"""ZED X Nano (wrist camera) over raw V4L2, without the ZED SDK.

ZED SDK >= 5.3 is required to open a ZED X Nano, but Stereolabs only ships 5.3+ for JetPack
6/7. The ZED Link driver 1.4.2 for L4T 35.4.1 exposes the Nano's two AR0234 sensors as plain
V4L2 nodes (10-bit Bayer GRBG, 'BA10'). Each sensor is read by a v4l2-ctl child and gets
software auto-exposure, gray-world white balance, demosaicing and JPEG encoding. The capture
and pairing code is that of the single-camera server (scripts/zedx_nano_v4l2_stream.py); here
the frames go to the camera's persistent slots, and the exposure, gain and white balance carry
over to the next activation.
"""
import collections
import fcntl
import select
import subprocess
import threading
import time

import cv2
import numpy as np

from sources import EYES, CameraSource, FrameSlot

MODEL = 'ZED X Nano'
SERVER_VERSION = 4                           # /info schema of the old single-camera server
STEREO_TOLERANCE_NS = 12_000_000             # left/right capture stamps must agree within this (< half a frame)
BAYER_CODES = {
    # v4l2 'BA10' = SGRBG10 (row0 G R, row1 B G). OpenCV names by 2nd row, 2nd+3rd columns.
    'GB': cv2.COLOR_BayerGB2BGR,
    'GR': cv2.COLOR_BayerGR2BGR,
    'RG': cv2.COLOR_BayerRG2BGR,
    'BG': cv2.COLOR_BayerBG2BGR,
}
EXPOSURE_MIN, EXPOSURE_MAX = 28, 66000       # microseconds (device tree min_exp_time/max_exp_time)
GAIN_MIN, GAIN_MAX = 100, 1600               # device tree min/max_gain_val with gain_factor 100 (1x-16x)
FRAME_TIMEOUT_S = 3.0                        # no bytes from v4l2-ctl for this long -> restart the capture
MMAP_BUFFERS = 4                             # v4l2-ctl --stream-mmap buffer count
F_SETPIPE_SZ = 1031                          # fcntl.F_SETPIPE_SZ, which Python 3.8 does not export
PIPE_SIZE = 1 << 20                          # /proc/sys/fs/pipe-max-size: a 4.6 MB frame in 5 writes, not 72
STOP_TIMEOUT_S = 6.0                         # v4l2_set() at a session start may take up to 5 s


def v4l2_set(device, **ctrls):
    arg = ','.join('%s=%d' % (k, int(v)) for k, v in ctrls.items())
    return subprocess.run(['v4l2-ctl', '-d', device, '--set-ctrl=' + arg],
                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=5).returncode == 0


def _step_off(value, minimum):
    return value + 1 if value <= minimum else value - 1


def output_size(width, height, out_width):
    if out_width and out_width < width:
        return [out_width, int(round(height * out_width / width))]
    return [width, height]


class NanoSensor(threading.Thread):
    """Captures one sensor node through v4l2-ctl and publishes JPEGs into `slot`.

    `stats` is the camera's dict for this eye. It outlives the sensor, so the counters and the
    auto-exposure and white-balance state continue with the next activation.
    """

    def __init__(self, device, label, slot, stats, width, height, fps, out_width, quality, bayer, ae, ae_target, wb):
        super().__init__(daemon=True, name='capture-' + label)
        self.device, self.width, self.height, self.fps = device, width, height, fps
        self.out_width, self.quality, self.bayer = out_width, quality, BAYER_CODES[bayer]
        self.ae, self.ae_target, self.wb = ae, ae_target, wb
        self.label, self.slot, self.stats = label, slot, stats
        self.exposure, self.gain = stats['exposure_us'], stats['gain']
        self.wb_gains = np.array(stats['wb_gains_bgr'], dtype=np.float64)
        self.stop_event = threading.Event()
        self.proc = None
        self.error = ''          # why the capture is down right now; cleared by the next frame
        self.processed = 0

    @property
    def output_size(self):
        return output_size(self.width, self.height, self.out_width)

    def stop(self):
        """Stop capturing and release the device now: kill the v4l2-ctl child instead of waiting for a frame."""
        self.stop_event.set()
        proc = self.proc
        if proc is not None and proc.poll() is None:
            proc.kill()

    # -- capture --------------------------------------------------------------------------------
    def run(self):
        frame_bytes = self.width * self.height * 2
        buf = bytearray(frame_bytes)
        max_exposure = min(EXPOSURE_MAX, int(0.95 * 1e6 / self.fps))
        while not self.stop_event.is_set():
            error = self._capture_session(buf, frame_bytes, max_exposure)
            if self.stop_event.is_set():
                break
            self.error = error[-300:] or 'capture stopped'
            self.stats['restarts'] += 1
            self.stats['last_error'] = self.error
            self.stats['fps'] = 0.0
            self.stop_event.wait(1.0)
        self.stats['fps'] = 0.0

    def _capture_session(self, buf, frame_bytes, max_exposure):
        """Run one v4l2-ctl child until it fails or stalls; returns a diagnostic string."""
        proc = None
        stderr_tail = collections.deque(maxlen=8)
        try:
            # Exposure and gain written while the sensor is idle never reach it, and the driver skips
            # writes of an unchanged value. So park them one step off here and write the real values
            # once frames flow; otherwise a restart keeps the sensor's defaults while AE believes otherwise.
            v4l2_set(self.device, frame_rate=self.fps * 1000000, exposure=_step_off(self.exposure, EXPOSURE_MIN),
                     gain=_step_off(self.gain, GAIN_MIN))
            cmd = ['v4l2-ctl', '-d', self.device,
                   '--set-fmt-video=width=%d,height=%d,pixelformat=BA10' % (self.width, self.height),
                   '--stream-mmap=%d' % MMAP_BUFFERS, '--stream-to=-']
            proc = self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
            try:
                fcntl.fcntl(proc.stdout.fileno(), F_SETPIPE_SZ, PIPE_SIZE)
            except OSError:
                pass   # keep the default 64 KiB pipe
            # v4l2-ctl prints a '<' per dequeued buffer and an fps line every second on stderr even
            # with --stream-to=-; drain it (bounded) or the child blocks once the pipe is full.
            threading.Thread(target=self._drain_stderr, args=(proc, stderr_tail), daemon=True).start()
            t_win, n_win = time.time(), 0
            first, skip = True, 0
            while not self.stop_event.is_set():
                if not self._read_frame(proc, buf, frame_bytes):
                    return 'capture ended or stalled: ' + b''.join(stderr_tail).decode(errors='ignore').strip()
                if first:
                    first = False
                    v4l2_set(self.device, exposure=self.exposure, gain=self.gain)
                    # This frame and those already queued in the driver's buffers were exposed with the
                    # sensor's defaults (dark): drop them rather than publish them or let AE react to them.
                    skip = 1 + MMAP_BUFFERS
                if skip:
                    skip -= 1
                else:
                    # v4l2-ctl writes each frame right after dequeuing it, so the end of the pipe
                    # transfer is the closest stamp to sensor readout we have.
                    self.process(buf, max_exposure, time.time_ns(), time.monotonic_ns())
                n_win += 1
                if time.time() - t_win >= 2.0:
                    self.stats['fps'] = round(n_win / (time.time() - t_win), 1)
                    t_win, n_win = time.time(), 0
            return ''
        except Exception as exc:  # v4l2-ctl missing/timeout, OpenCV errors, ...: restart instead of dying
            return 'capture error: %r; %s' % (exc, b''.join(stderr_tail).decode(errors='ignore').strip())
        finally:
            if proc is not None:
                proc.kill()
                proc.wait()
            self.proc = None

    @staticmethod
    def _drain_stderr(proc, tail):
        try:
            while True:
                chunk = proc.stderr.read1(256) if hasattr(proc.stderr, 'read1') else proc.stderr.read(256)
                if not chunk:
                    return
                tail.append(chunk)
        except Exception:
            return

    def _read_frame(self, proc, buf, frame_bytes):
        """Fill buf with one frame; False on EOF or when no bytes arrive for FRAME_TIMEOUT_S."""
        mv, got = memoryview(buf), 0
        fd = proc.stdout.fileno()
        while got < frame_bytes:
            ready, _, _ = select.select([fd], [], [], FRAME_TIMEOUT_S)
            if not ready:
                return False
            n = proc.stdout.readinto(mv[got:])
            if not n:
                return False
            got += n
        return True

    # -- processing -----------------------------------------------------------------------------
    def process(self, buf, max_exposure, capture_ns, capture_mono_ns):
        raw = np.frombuffer(buf, dtype=np.uint16).reshape(self.height, self.width)
        # T19x VI stores RAW10 left-aligned in 16-bit words: keep the top 8 bits.
        img8 = (raw >> 8).astype(np.uint8)
        bgr = cv2.cvtColor(img8, self.bayer)
        if self.out_width and self.out_width < self.width:
            bgr = cv2.resize(bgr, tuple(self.output_size), interpolation=cv2.INTER_AREA)
        if self.wb:
            means = bgr.reshape(-1, 3)[::97].mean(axis=0) + 1e-3
            target = np.clip(means[1] / means, 0.4, 3.0)
            self.wb_gains = 0.9 * self.wb_gains + 0.1 * target
            bgr = cv2.multiply(bgr, (float(self.wb_gains[0]), float(self.wb_gains[1]), float(self.wb_gains[2]), 0.0))
            self.stats['wb_gains_bgr'] = [round(float(g), 3) for g in self.wb_gains]
        ok, jpg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            return
        self.stats['frames'] = self.slot.publish(jpg.tobytes(), capture_ns, capture_mono_ns)
        self.error = ''
        self.processed += 1
        if self.processed % 4 == 0:
            self.auto_exposure(img8, max_exposure)

    def auto_exposure(self, img8, max_exposure):
        # GRBG mosaic: phase (0,0) is the G of the G/R rows, the luminance proxy the target is tuned
        # for; clipping is judged on the brightest of all four colour phases.
        phases = [img8[r::8, c::8] for r in (0, 1) for c in (0, 1)]
        mean = float(phases[0].mean())
        clipped = max(float((p >= 250).mean()) for p in phases)
        self.stats['mean'] = round(mean, 1)
        if not self.ae:
            return
        ratio = self.ae_target / max(mean, 1.0)
        if clipped > 0.10:            # large saturated areas: back off even if the mean is low
            ratio = min(ratio, 0.8)
        elif clipped > 0.03:          # some highlights: do not push the exposure up any further
            ratio = min(ratio, 1.0)
        if 0.93 < ratio < 1.07:
            return
        ratio = float(np.clip(ratio, 0.5, 2.0)) ** 0.7  # damped step
        total = self.exposure * (self.gain / 100.0) * ratio
        exposure = int(np.clip(total, EXPOSURE_MIN, max_exposure))
        gain = int(np.clip(round(100.0 * total / exposure), GAIN_MIN, GAIN_MAX))
        if exposure == self.exposure and gain == self.gain:
            return
        if v4l2_set(self.device, exposure=exposure, gain=gain):
            self.exposure, self.gain = exposure, gain
            self.stats['exposure_us'], self.stats['gain'] = exposure, gain


class NanoSession:
    """One activation of the Nano: a capture thread per sensor.

    The sensors restart their own v4l2-ctl child after a failure or stall, so the session only
    counts as dead when a capture thread itself died.
    """

    def __init__(self, sensors):
        self.sensors = sensors

    def start(self):
        for sensor in self.sensors:
            sensor.start()

    def stop(self, timeout=STOP_TIMEOUT_S):
        for sensor in self.sensors:
            sensor.stop()
        deadline = time.monotonic() + timeout
        for sensor in self.sensors:
            if sensor.ident is not None:
                sensor.join(max(0.0, deadline - time.monotonic()))
        return not any(sensor.is_alive() for sensor in self.sensors)

    @property
    def alive(self):
        return all(sensor.is_alive() for sensor in self.sensors)

    @property
    def error(self):
        return '; '.join('%s: %s' % (s.label, s.error) for s in self.sensors if s.error)


class NanoCamera(CameraSource):
    """The ZED X Nano: two V4L2 sensor nodes (left, right), paired by capture stamp."""
    backend = 'v4l2'

    def __init__(self, devices, serial, calibration, width=1920, height=1200, fps=30, out_width=960, quality=80,
                 bayer='GB', ae=True, ae_target=105.0, wb=True, exposure=8000, gain=200,
                 name='zedx_nano', role='wrist'):
        if len(devices) != 2:
            raise ValueError('the Nano needs two devices (left,right), got %r' % (devices,))
        self.eye_slots = {eye: FrameSlot() for eye in EYES}
        super().__init__(name, MODEL, serial, role, calibration, [self.eye_slots[eye] for eye in EYES])
        self.devices = dict(zip(EYES, devices))
        self.bayer, self.ae, self.wb = bayer, ae, wb
        self.capture = {'width': width, 'height': height, 'fps': fps}
        self.output = output_size(width, height, out_width)
        self.options = dict(width=width, height=height, fps=fps, out_width=out_width, quality=quality, bayer=bayer,
                            ae=ae, ae_target=ae_target, wb=wb)
        self.sensor_stats = {eye: {'device': device, 'frames': 0, 'fps': 0.0, 'mean': 0.0, 'exposure_us': exposure,
                                   'gain': gain, 'wb_gains_bgr': [1.0, 1.0, 1.0], 'restarts': 0, 'last_error': ''}
                             for eye, device in self.devices.items()}

    def open_session(self):
        return NanoSession([NanoSensor(self.devices[eye], eye, self.eye_slots[eye], self.sensor_stats[eye],
                                       **self.options) for eye in EYES])

    def info(self):
        return {
            'server_version': SERVER_VERSION,
            'model': self.model,
            'serial': self.serial,
            'capture': dict(self.capture, pixel_format='BA10'),
            'output': {'width': self.output[0], 'height': self.output[1], 'format': 'jpeg'},
            'sensors': dict(self.devices),
            'stereo': {'available': True, 'tolerance_us': STEREO_TOLERANCE_NS // 1000,
                       'note': 'left/right labels follow the calibration: on the Nano the driver\'s video2 is the right sensor'},
            'processing': {'auto_exposure': self.ae, 'white_balance': self.wb, 'bayer': self.bayer,
                           'rectified': False, 'isp': False},
        }

    def latest(self, eye, last_id, timeout=1.0):
        return self.eye_slots[eye].latest(last_id, timeout)

    def stereo_pairs(self, alive, tolerance_ns=STEREO_TOLERANCE_NS):
        """Pairs of a left frame and the right frame captured within tolerance of it (the old stereo_parts())."""
        left, right = self.eye_slots['left'], self.eye_slots['right']
        last_left = 0
        while alive():
            ljpg, lmeta = left.latest(last_left, timeout=1.0)
            if ljpg is None or lmeta['frame_id'] == last_left:
                yield None
                continue
            last_left = lmeta['frame_id']
            rjpg, rmeta = right.latest(-1, timeout=0)
            deadline = time.monotonic() + 0.04
            while (rjpg is None or lmeta['capture_mono_ns'] - rmeta['capture_mono_ns'] > tolerance_ns) and time.monotonic() < deadline:
                rjpg, rmeta = right.latest(rmeta['frame_id'], timeout=0.04)  # the right frame is still in flight
            if rjpg is None:
                continue
            if abs(lmeta['capture_mono_ns'] - rmeta['capture_mono_ns']) > tolerance_ns:
                continue  # no partner for this left frame; skip it rather than pair mismatched frames
            yield ljpg, lmeta, rjpg, rmeta

    def stats(self):
        sensors = {eye: dict(stats) for eye, stats in self.sensor_stats.items()}
        errors = ['%s: %s' % (eye, s['last_error']) for eye, s in sensors.items() if s['last_error']]
        return {'fps': min(s['fps'] for s in sensors.values()), 'frames': self.eye_slots['left'].frame_id,
                'restarts': self.restarts + sum(s['restarts'] for s in sensors.values()),
                'last_error': self.last_error or '; '.join(errors), 'sensors': sensors}

    def legacy_stats(self):
        """/stats of the old server: {label: sensor stats}."""
        return {eye: dict(stats) for eye, stats in self.sensor_stats.items()}
