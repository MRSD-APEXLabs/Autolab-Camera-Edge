"""ZED X (base camera) through the ZED SDK: hardware-synchronized, unrectified JPEG pairs.

The SDK opens the ZED X at SVGA 960x600 @ 60 fps with depth disabled (the Thor computes depth).
Both eyes come from one grab and share its capture stamp. The images are unrectified like the
Nano's, so the Thor rectifies both cameras the same way from the factory .conf file.
The SDK runs in a child process (zedx_capture.py): pyzed holds the GIL while it opens and closes
the camera, which in the hub's own interpreter stalled every request for ~5 s per switch. A
child also contains the SDK's memory and can be killed when it hangs.
"""
import collections
import fcntl
import json
import logging
import os
import select
import signal
import subprocess
import sys
import threading
import time

from sources import EYES, CameraSource, FrameSlot
from zedx_capture import (CAPTURE_FPS, CLOSED, ERROR, HEADER, HEIGHT, MAGIC, MAX_RECORD, OPENED, PAIR, PAIR_KIND,
                          STATS, WIDTH)

MODEL = 'ZED X'
SERVER_VERSION = 4                  # /info schema of the old single-camera server
CAPTURE_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'zedx_capture.py')
GRAB_ERROR_TIMEOUT_S = 3.0          # grab errors for this long end the capture; the hub reopens the camera
OPEN_TIMEOUT_S = 30.0               # an open (~5 s) that takes longer than this is a hung SDK: kill it
FRAME_TIMEOUT_S = 5.0               # no pair for this long after the open: kill the capture process
KILL_GRACE_S = 3.0                  # SIGTERM first (the camera is closed cleanly), SIGKILL after this
EXIT_TIMEOUT_S = 10.0               # the SDK's teardown after the camera closed (~0.6 s), then SIGKILL
STOP_TIMEOUT_S = 15.0               # the SDK cannot abort an open() in progress (~5 s)
F_SETPIPE_SZ, PIPE_SIZE = 1031, 1 << 20   # as for the Nano's v4l2-ctl: a ~150 kB pair in one write

log = logging.getLogger('camera_hub')


class ZedXSession(threading.Thread):
    """One open of the ZED X: runs zedx_capture.py and publishes the pairs it sends.

    The process is started in its own session so a Ctrl+C in the hub's terminal does not reach
    it; the hub stops it with SIGTERM, and it exits by itself once the hub's end of the pipe closes.
    """

    def __init__(self, camera):
        super().__init__(daemon=True, name='capture-' + camera.name)
        self.camera = camera
        self.stop_event = threading.Event()
        self.released = threading.Event()   # the camera is closed (or the process is gone)
        self.error = ''
        self.proc = None
        self.stderr_tail = collections.deque(maxlen=8)
        self._opened = False
        self._deadline = 0.0
        self._term_at = None

    @property
    def alive(self):
        return self.is_alive()

    def stop(self, timeout=STOP_TIMEOUT_S):
        """Ask the capture process to close the camera; kill it if it has not within `timeout`.

        Returns once the camera is closed. The process exits in the background.
        """
        self.stop_event.set()
        self._signal(signal.SIGTERM)
        if self.ident is None:
            return True
        if not self.released.wait(timeout):
            self._signal(signal.SIGKILL)
            self.released.wait(1.0)
        return self.released.is_set()

    def _signal(self, signum):
        proc = self.proc
        if proc is not None and proc.poll() is None:
            try:
                proc.send_signal(signum)
            except OSError:
                pass

    def run(self):
        try:
            self._run()
        except Exception as exc:  # pipe or protocol failure: end the session, the hub reopens the camera
            self.error = self.error or 'ZED X capture error: %r' % exc
        finally:
            proc = self.proc
            if proc is not None:
                proc.stdout.close()        # its next write fails, so a live process closes the camera and exits
                if proc.poll() is None:
                    self._signal(signal.SIGTERM)
                    try:
                        proc.wait(KILL_GRACE_S)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                proc.wait()
            self.camera.update_stats(fps=0.0, grab_fps=0.0)
            self.released.set()

    def _run(self):
        camera = self.camera
        self._deadline = time.monotonic() + camera.open_timeout
        proc = self.proc = subprocess.Popen(camera.capture_command(), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE, bufsize=0, start_new_session=True)
        if self.stop_event.is_set():
            self._signal(signal.SIGTERM)   # stop() came before the process existed
        try:
            fcntl.fcntl(proc.stdout.fileno(), F_SETPIPE_SZ, PIPE_SIZE)
        except OSError:
            pass   # keep the default 64 KiB pipe
        threading.Thread(target=self._drain_stderr, args=(proc, self.stderr_tail), daemon=True).start()
        while True:
            record = self._read_record(proc)
            if record is None:
                break
            kind, body = record
            if kind == PAIR_KIND:
                capture_ns, capture_mono_ns, left_length = PAIR.unpack_from(body)
                left_end = PAIR.size + left_length
                self._deadline = time.monotonic() + camera.frame_timeout
                if not self.stop_event.is_set():
                    camera.pairs.publish((bytes(body[PAIR.size:left_end]), bytes(body[left_end:])),
                                         capture_ns, capture_mono_ns)
            elif kind == STATS:
                camera.update_stats(**json.loads(body.decode()))
            elif kind == OPENED:
                self._opened = True
                self._deadline = time.monotonic() + camera.frame_timeout
                log.info('ZED X SN%d opened in %.1f s', camera.serial, json.loads(body.decode())['open_s'])
            elif kind == ERROR:
                self.error = body.decode('utf-8', 'replace')[-300:]
            elif kind == CLOSED:
                self.released.set()
            self._watchdog()
        try:
            code = proc.wait(EXIT_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            code = proc.wait()
        if not self.error and not self.stop_event.is_set():
            self.error = ('ZED X capture exited with code %s: %s'
                          % (code, b''.join(self.stderr_tail).decode(errors='ignore').strip()))[-300:]

    def _read_record(self, proc):
        """(kind, body) of the next record, or None at EOF."""
        header = self._read_exact(proc, HEADER.size)
        if header is None:
            return None
        magic, kind, length = HEADER.unpack(header)
        if magic != MAGIC or length > MAX_RECORD:
            raise ValueError('bad record header %r from zedx_capture.py' % bytes(header))
        body = self._read_exact(proc, length)
        return None if body is None else (kind, body)

    def _read_exact(self, proc, n):
        """n bytes from the capture process (None at EOF). Waits in short steps so the watchdog runs."""
        buf = bytearray(n)
        view, got = memoryview(buf), 0
        fd = proc.stdout.fileno()
        while got < n:
            ready, _, _ = select.select([fd], [], [], 0.5)
            if not ready:
                self._watchdog()
                continue
            count = proc.stdout.readinto(view[got:])
            if not count:
                return None
            got += count
        return buf

    def _watchdog(self):
        """A process that opens or sends pairs too slowly is hung in the SDK: SIGTERM, then SIGKILL."""
        now = time.monotonic()
        if self._term_at is not None:
            if now - self._term_at > KILL_GRACE_S:
                self._signal(signal.SIGKILL)
            return
        if now > self._deadline and not self.stop_event.is_set():
            if self._opened:
                self.error = 'ZED X sent no frames for %.1f s' % self.camera.frame_timeout
            else:
                self.error = 'ZED X open did not finish within %.0f s' % self.camera.open_timeout
            self._term_at = now
            self._signal(signal.SIGTERM)

    @staticmethod
    def _drain_stderr(proc, tail):
        try:
            while True:
                chunk = proc.stderr.read(256)
                if not chunk:
                    return
                tail.append(chunk)
        except Exception:
            return


class ZedXCamera(CameraSource):
    """The ZED X: one grab gives both eyes, so a slot holds (left_jpeg, right_jpeg) pairs."""
    backend = 'zed_sdk'

    def __init__(self, serial, calibration, stream_fps=30, quality=80, name='zedx', role='base', cv_threads=1,
                 grab_error_timeout=GRAB_ERROR_TIMEOUT_S, open_timeout=OPEN_TIMEOUT_S, frame_timeout=FRAME_TIMEOUT_S):
        self.pairs = FrameSlot()
        super().__init__(name, MODEL, serial, role, calibration, [self.pairs])
        self.stream_fps = min(float(stream_fps), CAPTURE_FPS)
        self.quality, self.cv_threads = quality, cv_threads
        self.grab_error_timeout, self.open_timeout = grab_error_timeout, open_timeout
        # The capture process reports grab errors itself; the watchdog is for an SDK call that hangs.
        self.frame_timeout = max(frame_timeout, grab_error_timeout + 1.0, 3.0 / self.stream_fps)
        self._stats = {'fps': 0.0, 'grab_fps': 0.0, 'encode_ms': 0.0}

    def open_session(self):
        return ZedXSession(self)

    def capture_command(self):
        return [sys.executable, CAPTURE_SCRIPT, '--serial', str(self.serial), '--stream-fps', repr(self.stream_fps),
                '--quality', str(self.quality), '--grab-error-timeout', repr(float(self.grab_error_timeout)),
                '--cv-threads', str(self.cv_threads)]

    def update_stats(self, **values):
        self._stats.update(values)

    def info(self):
        stream_fps = int(self.stream_fps) if self.stream_fps == int(self.stream_fps) else self.stream_fps
        return {
            'server_version': SERVER_VERSION,
            'model': self.model,
            'serial': self.serial,
            'capture': {'width': WIDTH, 'height': HEIGHT, 'fps': CAPTURE_FPS, 'pixel_format': 'zed_sdk'},
            'output': {'width': WIDTH, 'height': HEIGHT, 'format': 'jpeg'},
            'sensors': {'left': 'LEFT_UNRECTIFIED', 'right': 'RIGHT_UNRECTIFIED'},
            'stereo': {'available': True, 'tolerance_us': 0,
                       'note': 'hardware-synchronized: both eyes come from one grab and share its capture stamp'},
            'processing': {'isp': True, 'rectified': False, 'stream_fps': stream_fps},
        }

    def latest(self, eye, last_id, timeout=1.0):
        pair, meta = self.pairs.latest(last_id, timeout)
        return (None if pair is None else pair[EYES.index(eye)]), meta

    def stereo_pairs(self, alive):
        last = 0
        while alive():
            pair, meta = self.pairs.latest(last, timeout=1.0)
            if pair is None or meta['frame_id'] == last:
                yield None
                continue
            last = meta['frame_id']
            yield pair[0], meta, pair[1], meta

    def stats(self):
        return dict(self._stats, frames=self.pairs.frame_id, stream_fps=self.stream_fps)
