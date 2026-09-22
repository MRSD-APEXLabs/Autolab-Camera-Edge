import fcntl
import os
import sys
import threading
import time

import cv2
import numpy as np
import pytest

import nano_source
from hub_fakes import wait_for
from nano_source import NanoCamera, NanoSensor
from sources import Calibration, FrameSlot

F_GETPIPE_SZ = 1032

# Stands in for v4l2-ctl: --set-ctrl succeeds; streaming writes uniform 16-bit Bayer frames to stdout
# whose level rises by 10 per frame (FAKE_V4L2_MODE=stream), fails like a busy device (fail) or sends
# a few frames and hangs (stall).
FAKE_V4L2 = r'''#!%s
import os, sys, time
args = sys.argv[1:]
if any(a.startswith('--set-ctrl') for a in args):
    with open(os.path.join(os.environ['FAKE_V4L2_PIDS'], 'ctrls.log'), 'a') as log:
        log.write(' '.join(args) + '\n')
    sys.exit(0)
open(os.path.join(os.environ['FAKE_V4L2_PIDS'], str(os.getpid())), 'w').close()
mode = os.environ.get('FAKE_V4L2_MODE', 'stream')
if mode == 'fail':
    sys.stderr.write('VIDIOC_S_FMT: failed: Device or resource busy\n')
    sys.exit(1)
fmt = [a for a in args if a.startswith('--set-fmt-video=')][0].split('=', 1)[1]
fields = dict(kv.split('=') for kv in fmt.split(','))
pixels = int(fields['width']) * int(fields['height'])
level = 20
for n in range(10 ** 9):
    sys.stdout.buffer.write(bytes(bytearray([0, level]) * pixels))
    sys.stdout.buffer.flush()
    sys.stderr.write('<')
    sys.stderr.flush()
    level = min(level + 10, 250)
    if mode == 'stall' and n == 7:
        time.sleep(3600)
    time.sleep(0.02)
'''


@pytest.fixture
def fake_v4l2(tmp_path, monkeypatch):
    """Put the fake v4l2-ctl first on PATH; returns (set_mode, pids of the streaming children)."""
    bin_dir, pid_dir = tmp_path / 'bin', tmp_path / 'pids'
    bin_dir.mkdir()
    pid_dir.mkdir()
    script = bin_dir / 'v4l2-ctl'
    script.write_text(FAKE_V4L2 % sys.executable)
    script.chmod(0o755)
    monkeypatch.setenv('PATH', str(bin_dir) + os.pathsep + os.environ['PATH'])
    monkeypatch.setenv('FAKE_V4L2_PIDS', str(pid_dir))

    def set_mode(mode):
        monkeypatch.setenv('FAKE_V4L2_MODE', mode)

    def pids():
        return [int(name) for name in os.listdir(str(pid_dir)) if name.isdigit()]
    return set_mode, pids


def small_camera(tmp_path):
    return NanoCamera(['/dev/fakeL', '/dev/fakeR'], 99292912, Calibration(99292912, str(tmp_path), 'FHD1200'),
                      width=64, height=40, fps=30, out_width=32)


def process_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_session_streams_both_eyes_and_stop_kills_the_children(tmp_path, fake_v4l2):
    set_mode, pids = fake_v4l2
    set_mode('stream')
    camera = small_camera(tmp_path)
    session = camera.open_session()
    session.start()
    try:
        assert wait_for(lambda: camera.eye_slots['left'].frame_id >= 3 and camera.eye_slots['right'].frame_id >= 3, 10)
        assert session.alive and session.error == ''
        for eye in ('left', 'right'):
            jpg, meta = camera.latest(eye, -1)
            image = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            assert image.shape == (20, 32, 3) and meta['capture_mono_ns'] > 0
        assert camera.stats()['frames'] >= 3 and camera.stats()['sensors']['right']['device'] == '/dev/fakeR'
        assert len(pids()) == 2 and all(process_alive(pid) for pid in pids())
        with open('/proc/sys/fs/pipe-max-size') as f:
            if int(f.read()) >= nano_source.PIPE_SIZE:   # a frame crosses the pipe in few writes
                proc = session.sensors[0].proc
                assert fcntl.fcntl(proc.stdout.fileno(), F_GETPIPE_SZ) == nano_source.PIPE_SIZE
    finally:
        t0 = time.monotonic()
        assert session.stop()
    assert time.monotonic() - t0 < 1.5
    assert not any(process_alive(pid) for pid in pids())


def test_exposure_and_gain_are_written_again_once_frames_flow(tmp_path, fake_v4l2):
    set_mode, _ = fake_v4l2
    set_mode('stream')
    camera = small_camera(tmp_path)
    camera.sensor_stats['left'].update(exposure_us=31666, gain=1041)   # carried over from the last activation
    session = camera.open_session()
    session.start()
    log = tmp_path / 'pids' / 'ctrls.log'
    try:
        assert wait_for(lambda: log.exists() and log.read_text().count('/dev/fakeL') >= 2, 10)
    finally:
        session.stop()
    left = [line for line in log.read_text().splitlines() if '/dev/fakeL' in line]
    # the idle sensor ignores writes and the driver skips unchanged values: park one step off, then set
    assert left[0] == '-d /dev/fakeL --set-ctrl=frame_rate=30000000,exposure=31665,gain=1040'
    assert left[1] == '-d /dev/fakeL --set-ctrl=exposure=31666,gain=1041'


def test_frames_exposed_before_the_controls_took_effect_are_dropped(tmp_path, fake_v4l2):
    set_mode, _ = fake_v4l2
    set_mode('stream')
    camera = small_camera(tmp_path)
    session = camera.open_session()
    session.start()
    try:
        assert wait_for(lambda: camera.eye_slots['left'].frame_id >= 1, 10)
        jpg, meta = camera.eye_slots['left'].latest(0, timeout=0)
    finally:
        session.stop()
    # the first frame and the MMAP_BUFFERS frames queued behind it (levels 20..60) are not published
    image = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_GRAYSCALE)
    if meta['frame_id'] == 1:
        assert abs(float(image.mean()) - (20 + 10 * (1 + nano_source.MMAP_BUFFERS))) < 3
    assert float(image.mean()) > 20 + 10 * nano_source.MMAP_BUFFERS + 3


def test_failing_device_is_reported_and_retried(tmp_path, fake_v4l2):
    set_mode, pids = fake_v4l2
    set_mode('fail')
    camera = small_camera(tmp_path)
    session = camera.open_session()
    session.start()
    try:
        assert wait_for(lambda: 'Device or resource busy' in session.error, 10)
        assert session.alive     # the sensors keep retrying on their own
        assert wait_for(lambda: camera.stats()['sensors']['left']['restarts'] >= 2, 10)
        assert 'left: capture ended or stalled' in camera.stats()['last_error']
        set_mode('stream')       # the device became free
        assert wait_for(lambda: camera.frames_since(time.monotonic() - 0.5) and session.error == '', 10)
    finally:
        session.stop()


def test_stalled_sensor_restarts_its_capture(tmp_path, fake_v4l2, monkeypatch):
    set_mode, pids = fake_v4l2
    set_mode('stall')
    monkeypatch.setattr(nano_source, 'FRAME_TIMEOUT_S', 0.3)
    camera = small_camera(tmp_path)
    session = camera.open_session()
    session.start()
    try:
        assert wait_for(lambda: len(pids()) >= 4, 10)   # each sensor started a second v4l2-ctl
        assert camera.eye_slots['left'].frame_id >= 1 and 'stalled' in camera.stats()['sensors']['left']['last_error']
    finally:
        session.stop()
    assert not any(process_alive(pid) for pid in pids())


def test_processing_white_balance_and_auto_exposure(monkeypatch):
    calls = []
    monkeypatch.setattr(nano_source, 'v4l2_set', lambda device, **ctrls: calls.append((device, ctrls)) or True)
    stats = {'device': '/dev/x', 'frames': 0, 'fps': 0.0, 'mean': 0.0, 'exposure_us': 8000, 'gain': 200,
             'wb_gains_bgr': [1.0, 1.0, 1.0], 'restarts': 0, 'last_error': ''}
    slot = FrameSlot()
    sensor = NanoSensor('/dev/x', 'left', slot, stats, 64, 40, 30, 32, 80, 'GB', True, 105.0, True)
    raw = np.full((40, 64), 40 << 8, np.uint16)        # a dark, uniform scene
    for i in range(4):
        sensor.process(bytearray(raw.tobytes()), 31666, 1000 + i, 2000 + i)
    assert slot.frame_id == 4 and stats['frames'] == 4 and slot.meta['capture_mono_ns'] == 2003
    image = cv2.imdecode(np.frombuffer(slot.payload, np.uint8), cv2.IMREAD_COLOR)
    assert image.shape == (20, 32, 3)
    # too dark: one damped step up, exposure first (capped at 95 % of the frame time), then gain
    assert len(calls) == 1 and calls[0][1]['exposure'] > 8000 and stats['exposure_us'] == calls[0][1]['exposure']
    assert stats['mean'] == 40.0
    # the next activation continues from the adjusted exposure and white balance
    again = NanoSensor('/dev/x', 'left', slot, stats, 64, 40, 30, 32, 80, 'GB', True, 105.0, True)
    assert again.exposure == stats['exposure_us'] and list(again.wb_gains) == stats['wb_gains_bgr']


def test_stereo_pairs_wait_for_the_partner_and_skip_mismatches():
    camera = NanoCamera(['/dev/a', '/dev/b'], 1, Calibration(1, '.', 'FHD1200'))
    left, right = camera.eye_slots['left'], camera.eye_slots['right']
    calls = []

    def alive():
        calls.append(1)
        return len(calls) <= 3
    pairs = camera.stereo_pairs(alive)
    ms = 1_000_000
    left.publish(b'L1', 1, 1000 * ms)
    right.publish(b'R1', 1, 1005 * ms)
    assert next(pairs)[::2] == (b'L1', b'R1')
    # right frame still in flight: the pairing waits for it (up to 40 ms)
    left.publish(b'L2', 2, 1033 * ms)
    threading.Timer(0.01, right.publish, args=(b'R2', 2, 1034 * ms)).start()
    ljpg, lmeta, rjpg, rmeta = next(pairs)
    assert (ljpg, rjpg) == (b'L2', b'R2') and lmeta['capture_mono_ns'] - rmeta['capture_mono_ns'] == -ms
    # no partner within 12 ms: the left frame is dropped, not paired with a mismatched right frame
    left.publish(b'L3', 3, 1066 * ms)
    right.publish(b'R3', 3, 1100 * ms)
    with pytest.raises(StopIteration):
        next(pairs)
