import json
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

import camera_hub
import zedx_capture
import zedx_source
from hub_fakes import CONF, Client, FakeNano, wait_for
from manager import ERROR, STREAMING, CameraHub
from sources import Calibration
from zedx_capture import OPENED, PAIR, PAIR_KIND, STATS
from zedx_source import ZedXCamera

FAKE_PYZED = str(Path(__file__).resolve().parent / 'fake_pyzed')


@pytest.fixture
def fake_sdk(monkeypatch):
    """The fake pyzed.sl, imported afresh in this process; returns its Camera class."""
    monkeypatch.syspath_prepend(FAKE_PYZED)
    monkeypatch.delenv('FAKE_ZED', raising=False)
    monkeypatch.delenv('FAKE_ZED_LOG', raising=False)
    for name in ('pyzed', 'pyzed.sl'):
        monkeypatch.delitem(sys.modules, name, raising=False)
    import pyzed.sl as sl
    return sl


@pytest.fixture
def zed_process(monkeypatch, tmp_path):
    """Make zedx_capture.py processes load the fake SDK; returns (configure(**attrs), log path)."""
    monkeypatch.setenv('PYTHONPATH', os.pathsep.join(p for p in (FAKE_PYZED, os.environ.get('PYTHONPATH')) if p))
    log = str(tmp_path / 'zed.log')
    monkeypatch.setenv('FAKE_ZED_LOG', log)
    monkeypatch.setenv('FAKE_ZED', '{}')

    def configure(**attrs):
        monkeypatch.setenv('FAKE_ZED', json.dumps(attrs))
    return configure, log


def logged(path, event):
    """Pids of the capture processes that logged `event` (open, close, hang) in the fake SDK's log."""
    try:
        with open(path) as f:
            return [int(line.split()[1]) for line in f if line.split()[0] == event]
    except FileNotFoundError:
        return []


def make_camera(tmp_path, **kwargs):
    conf = tmp_path / 'SN42757821.conf'
    conf.write_text(CONF)
    return ZedXCamera(42757821, Calibration(42757821, str(tmp_path), 'SVGA', str(conf)), **kwargs)


class Capture:
    """zedx_capture.capture() in a thread, collecting its records."""

    def __init__(self, sl, stream_fps=30, grab_error_timeout=3.0, stopped=False):
        self.records, self.result = [], None
        self.stop = threading.Event()
        if stopped:
            self.stop.set()
        self.thread = threading.Thread(target=self._run, args=(sl, stream_fps, grab_error_timeout), daemon=True)
        self.thread.start()

    def _run(self, sl, stream_fps, grab_error_timeout):
        self.result = zedx_capture.capture(sl, 42757821, stream_fps, 80, grab_error_timeout,
                                           lambda kind, *parts: self.records.append((kind, b''.join(parts))),
                                           self.stop.is_set)

    def pairs(self):
        """[(capture_ns, capture_mono_ns, left_jpeg, right_jpeg)]"""
        pairs = []
        for kind, body in list(self.records):
            if kind == PAIR_KIND:
                capture_ns, capture_mono_ns, left_length = PAIR.unpack_from(body)
                pairs.append((capture_ns, capture_mono_ns, body[PAIR.size:PAIR.size + left_length],
                              body[PAIR.size + left_length:]))
        return pairs

    def finish(self, timeout=5):
        self.stop.set()
        self.thread.join(timeout)
        assert not self.thread.is_alive()
        return self.result


def decode(jpg):
    return cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)


# -- zedx_capture.capture(), in this process ---------------------------------------------------------
def test_capture_sends_decimated_synchronized_pairs(fake_sdk):
    capture = Capture(fake_sdk)
    try:
        assert wait_for(lambda: any(kind == STATS for kind, _ in capture.records), 5)
        zed = fake_sdk.Camera.instances[-1]
        init = zed.init
        assert (init.camera_resolution, init.camera_fps, init.depth_mode, init.sdk_verbose, init.serial) == \
            ('SVGA', 60, 'NONE', 0, 42757821)
        assert zed.runtime.enable_depth is False
        assert capture.records[0][0] == OPENED and 'open_s' in json.loads(capture.records[0][1].decode())
        pairs = capture.pairs()
        # 30 of 60 fps: only the published frames are retrieved (both eyes each)
        assert abs(zed.grabs / len(pairs) - 2.0) < 0.3
        assert zed.retrieved[:4] == ['left', 'right', 'left', 'right'] and len(zed.retrieved) <= 2 * len(pairs) + 2
        capture_ns, capture_mono_ns, ljpg, rjpg = pairs[-1]
        assert capture_ns <= time.time_ns() - 30_000_000
        assert abs(time.monotonic_ns() - capture_mono_ns - 30e6) < 60e6
        gaps = np.diff([p[1] for p in pairs]) / 1e6
        assert abs(np.median(gaps) - 33.3) < 3
        left, right = decode(ljpg), decode(rjpg)
        assert left.shape == right.shape == (600, 960, 3) and abs(left.mean() - 60) < 3 and abs(right.mean() - 190) < 3
        stats = json.loads([body for kind, body in capture.records if kind == STATS][0].decode())
        assert 25 <= stats['fps'] <= 35 and 50 <= stats['grab_fps'] <= 70 and stats['encode_ms'] > 0
    finally:
        assert capture.finish() == ''
    assert zed.closed


def test_capture_stream_fps_decimation(fake_sdk):
    capture = Capture(fake_sdk, stream_fps=15)
    try:
        assert wait_for(lambda: len(capture.pairs()) >= 8, 10)
        assert abs(fake_sdk.Camera.instances[-1].grabs / len(capture.pairs()) - 4.0) < 0.6
    finally:
        capture.finish()


def test_capture_open_failure(fake_sdk, monkeypatch):
    monkeypatch.setattr(fake_sdk.Camera, 'open_results', ['CAMERA_NOT_DETECTED'])
    capture = Capture(fake_sdk)
    capture.thread.join(5)
    assert 'open failed' in capture.result and 'CAMERA_NOT_DETECTED' in capture.result
    assert fake_sdk.Camera.instances[-1].closed and capture.records == []


def test_capture_wrong_resolution(fake_sdk, monkeypatch):
    monkeypatch.setattr(fake_sdk.Camera, 'resolution', (1920, 1200))
    capture = Capture(fake_sdk)
    capture.thread.join(5)
    assert capture.result == 'ZED X opened at 1920x1200 instead of 960x600'


def test_capture_stopped_during_the_open_sends_nothing(fake_sdk):
    capture = Capture(fake_sdk, stopped=True)     # SIGTERM came while open() ran
    assert capture.finish() == ''
    assert capture.records == [] and fake_sdk.Camera.instances[-1].closed


def test_capture_grab_errors_end_it_only_when_they_persist(fake_sdk):
    capture = Capture(fake_sdk, grab_error_timeout=0.4)
    assert wait_for(lambda: len(capture.pairs()) >= 2, 5)
    now = time.monotonic()
    fake_sdk.Camera.failing = (now, now + 0.15)           # a short hiccup
    time.sleep(0.4)
    assert capture.thread.is_alive()
    first = len(capture.pairs())
    assert wait_for(lambda: len(capture.pairs()) > first, 5)
    now = time.monotonic()
    fake_sdk.Camera.failing = (now, now + 60)
    capture.thread.join(3)
    assert time.monotonic() - now < 1.5
    assert capture.result.startswith('ZED X grab failing for') and 'CAMERA_REBOOTING' in capture.result
    assert fake_sdk.Camera.instances[-1].closed


def test_wall_clock_step_back_keeps_the_pairs_coming(fake_sdk, monkeypatch):
    """NTP steps the wall clock back 60 s; the SDK stamps follow it. Decimation and monotonic stamps don't."""
    capture = Capture(fake_sdk)
    try:
        assert wait_for(lambda: len(capture.pairs()) >= 5, 5)
        real_time_ns = time.time_ns
        monkeypatch.setattr(time, 'time_ns', lambda: real_time_ns() - 60_000_000_000)
        before = len(capture.pairs())
        time.sleep(0.5)
        pairs = capture.pairs()
        assert len(pairs) - before >= 12
        gaps = np.diff([p[1] for p in pairs]) / 1e6
        assert gaps.min() > 20 and gaps.max() < 60           # CLOCK_MONOTONIC stamps: no jump, no stall
        assert pairs[-1][0] < real_time_ns() - 59e9          # the wall-clock stamp is the SDK's
    finally:
        capture.finish()


@pytest.mark.parametrize('offset_s', [-14 * 3600, 14 * 3600])
def test_image_stamps_off_the_wall_clock_keep_their_last_age(fake_sdk, offset_s):
    """If the SDK kept its clock offset from the open while the wall clock stepped (NTP at boot)."""
    capture = Capture(fake_sdk)
    try:
        assert wait_for(lambda: len(capture.pairs()) >= 5, 5)
        fake_sdk.Camera.clock_offset_ns = offset_s * 10 ** 9
        before = len(capture.pairs())
        time.sleep(0.5)
        pairs = capture.pairs()
        assert len(pairs) - before >= 12
        capture_ns, capture_mono_ns, _, _ = pairs[-1]
        assert abs(time.monotonic_ns() - capture_mono_ns - 30e6) < 60e6
        assert abs(time.time_ns() - capture_ns - 30e6) < 60e6
    finally:
        capture.finish()


def test_fake_open_holds_the_gil_like_pyzed(fake_sdk):
    """The premise of the responsiveness test below: in-process, the fake open stalls other threads."""
    ticks = []
    stop = threading.Event()

    def tick():
        while not stop.is_set():
            ticks.append(time.monotonic())
            time.sleep(0.005)
    thread = threading.Thread(target=tick)
    thread.start()
    time.sleep(0.05)
    fake_sdk.hold_gil(0.4)
    stop.set()
    thread.join()
    assert max(np.diff(ticks)) >= 0.35


# -- ZedXSession: zedx_capture.py processes ----------------------------------------------------------
def test_session_streams_through_the_capture_process(tmp_path, zed_process):
    configure, log = zed_process
    configure(noise=True, exit_hold_s=1.0)   # SDK output on stdout must not corrupt the records
    camera = make_camera(tmp_path)
    session = camera.open_session()
    session.start()
    try:
        assert wait_for(lambda: camera.pairs.frame_id >= 10, 15)
        (ljpg, rjpg), meta = camera.pairs.latest(-1)
        assert abs(time.monotonic_ns() - meta['capture_mono_ns'] - 30e6) < 100e6
        assert meta['capture_ns'] <= time.time_ns() - 30_000_000
        assert abs(decode(ljpg).mean() - 60) < 3 and abs(decode(rjpg).mean() - 190) < 3
        assert wait_for(lambda: camera.stats()['fps'] > 0, 5)
        assert 25 <= camera.stats()['fps'] <= 35 and 50 <= camera.stats()['grab_fps'] <= 70
        pid = session.proc.pid
        assert session.error == '' and session.alive
    finally:
        t0 = time.monotonic()
        assert session.stop()
    # stop() returns once the camera is closed, without waiting for the SDK's teardown at exit
    assert time.monotonic() - t0 < 0.8 and session.proc.poll() is None
    assert logged(log, 'close') == [pid]
    assert wait_for(lambda: not session.alive, 5) and session.proc.returncode == 0
    assert session.error == '' and camera.stats()['fps'] == 0.0


@pytest.mark.parametrize('config, error', [
    ({'open_results': ['CAMERA_NOT_DETECTED']}, 'ZED X SN42757821 open failed:'),
    ({'resolution': [1920, 1200]}, 'ZED X opened at 1920x1200 instead of 960x600'),
    ({'missing': True}, 'pyzed (ZED SDK Python API) is not installed'),
    ({'open_results': ['NO_SUCH_CODE']}, "ZED X capture error: KeyError('NO_SUCH_CODE'"),
    ({'crash_after': 3}, 'ZED X capture exited with code 3'),
    ({'fail_after': 3}, 'ZED X grab failing for'),
])
def test_session_reports_why_the_capture_ended(tmp_path, zed_process, config, error):
    configure, _ = zed_process
    configure(**config)
    camera = make_camera(tmp_path, grab_error_timeout=0.3)
    session = camera.open_session()
    session.start()
    assert wait_for(lambda: not session.alive, 10)
    assert session.error.startswith(error), session.error
    assert session.stop()


def test_hung_sdk_is_killed(tmp_path, zed_process, monkeypatch):
    configure, log = zed_process
    configure(hang_after=5)
    monkeypatch.setattr(zedx_source, 'KILL_GRACE_S', 0.3)
    camera = make_camera(tmp_path, grab_error_timeout=0.2, frame_timeout=0.5)
    session = camera.open_session()
    session.start()
    assert wait_for(lambda: logged(log, 'hang'), 10)
    t0 = time.monotonic()
    assert wait_for(lambda: session.error != '', 5)
    assert session.error == 'ZED X sent no frames for 1.2 s'
    assert wait_for(lambda: not session.alive, 5) and time.monotonic() - t0 < 4
    assert session.proc.returncode == -9 and not logged(log, 'close')


def test_a_failure_in_the_hub_still_closes_the_camera(tmp_path, zed_process, monkeypatch):
    _, log = zed_process
    camera = make_camera(tmp_path)

    def publish(*args):
        raise RuntimeError('slot broken')
    monkeypatch.setattr(camera.pairs, 'publish', publish)
    session = camera.open_session()
    session.start()
    assert wait_for(lambda: not session.alive, 15)
    assert session.error == "ZED X capture error: RuntimeError('slot broken')"
    assert logged(log, 'close') == [session.proc.pid] and session.proc.returncode == 0


def test_garbage_on_the_pipe_is_a_protocol_error(tmp_path):
    read_fd, write_fd = os.pipe()
    os.write(write_fd, b'[ZED] not a record\n')
    os.close(write_fd)

    class Proc:
        stdout = os.fdopen(read_fd, 'rb', buffering=0)
    session = make_camera(tmp_path).open_session()
    with pytest.raises(ValueError, match='bad record header'):
        session._read_record(Proc)
    Proc.stdout.close()


def test_stop_during_the_open_waits_for_it_and_closes_the_camera(tmp_path, zed_process):
    configure, log = zed_process
    configure(open_hold_s=1.0)
    camera = make_camera(tmp_path)
    session = camera.open_session()
    session.start()
    assert wait_for(lambda: logged(log, 'open'), 10)
    assert session.stop()
    assert logged(log, 'close') and wait_for(lambda: session.proc.poll() == 0, 5)
    assert camera.pairs.frame_id == 0 and session.error == ''


def test_hub_answers_while_the_zedx_opens(tmp_path, zed_process):
    """pyzed holds the GIL through open(): in the hub's interpreter it froze every request for ~5 s."""
    configure, _ = zed_process
    configure(open_hold_s=1.5)
    (tmp_path / 'SN99292912.conf').write_text(CONF)
    hub = CameraHub([make_camera(tmp_path), FakeNano(tmp_path)], str(tmp_path / 'state.json'), default='zedx_nano',
                    tick_s=0.02)
    server = camera_hub.build_server(hub, '127.0.0.1', 0)
    hub.start()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    client = Client(server.server_address[1])
    latencies, stop = [], threading.Event()

    def poll():
        while not stop.is_set():
            t0 = time.monotonic()
            assert client.json('GET', '/time')[0] == 200
            latencies.append(time.monotonic() - t0)
            time.sleep(0.05)
    try:
        assert wait_for(lambda: hub.state == STREAMING)
        poller = threading.Thread(target=poll)
        poller.start()
        t0 = time.monotonic()
        code, reply = client.select('zedx')
        assert code == 200 and reply['status']['active'] == 'zedx' and time.monotonic() - t0 >= 1.5
        stop.set()
        poller.join(5)
        assert len(latencies) >= 15 and max(latencies) < 0.5
    finally:
        stop.set()
        hub.close()
        server.shutdown()
        server.server_close()


def test_hub_reopens_the_zedx_after_a_failed_open(tmp_path, zed_process):
    configure, log = zed_process
    configure(open_results=['CAMERA_NOT_DETECTED'])
    camera = make_camera(tmp_path)
    hub = CameraHub([camera], str(tmp_path / 'state.json'), default=None, retry_delays_s=(0.2,), tick_s=0.02)
    hub.start()
    try:
        assert hub.select('zedx') == 'error'
        assert hub.state == ERROR and 'CAMERA_NOT_DETECTED' in hub.error
        assert wait_for(lambda: hub.state == STREAMING, 10)
        assert len(logged(log, 'open')) == 2 and len(logged(log, 'close')) == 1
        assert hub.status()['cameras']['zedx']['restarts'] == 1
        assert hub.select(None) == 'ok'
        assert len(logged(log, 'close')) == 2
    finally:
        hub.close()


def test_capture_command(tmp_path):
    camera = make_camera(tmp_path, stream_fps=15, quality=70, cv_threads=2)
    command = camera.capture_command()
    assert command[:2] == [sys.executable, zedx_source.CAPTURE_SCRIPT] and os.path.exists(command[1])
    args = zedx_capture.parse_args(command[2:])
    assert (args.serial, args.stream_fps, args.quality, args.grab_error_timeout, args.cv_threads) == \
        (42757821, 15.0, 70, 3.0, 2)
    assert camera.frame_timeout == 5.0
