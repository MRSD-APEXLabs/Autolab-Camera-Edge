"""Stand-ins for the capture backends, plus a small MJPEG reader and HTTP helpers.

The fake cameras are the real ZedXCamera / NanoCamera classes with a synthetic capture
session, so the hub, the HTTP layer, pairing and the /info schemas run unchanged.
"""
import http.client
import json
import threading
import time

import cv2
import numpy as np

from nano_source import NanoCamera
from sources import Calibration
from zedx_source import ZedXCamera

CONF = """[LEFT_CAM_FHD1200]
fx=950
[LEFT_CAM_SVGA]
fx=475
[STEREO]
Baseline=18.01
"""


def jpeg(value, size=(8, 6)):
    return cv2.imencode('.jpg', np.full((size[1], size[0], 3), value, np.uint8))[1].tobytes()


class FakeSession(threading.Thread):
    """Publishes synthetic frames at `camera.fps`. The camera's knobs script failures:
    `fail_starts` sessions die at once, `start_delay` delays the first frame, `stall` pauses publishing."""

    def __init__(self, camera):
        super().__init__(daemon=True, name='fake-' + camera.name)
        self.camera = camera
        self.stop_event = threading.Event()
        self.error = ''
        self.stopped_at = None

    @property
    def alive(self):
        return self.is_alive()

    def stop(self, timeout=5.0):
        self.stop_event.set()
        if self.ident is not None:
            self.join(timeout)
        self.stopped_at = time.monotonic()
        return not self.is_alive()

    def run(self):
        camera = self.camera
        camera.sessions.append(self)
        if camera.fail_starts > 0:
            camera.fail_starts -= 1
            self.error = 'fake open failed'
            return
        if self.stop_event.wait(camera.start_delay):
            return
        while not self.stop_event.wait(1.0 / camera.fps):
            if not camera.stall.is_set():
                camera.publish_frame()


class FakeCameraMixin:
    def setup_fake(self, fps=50):
        self.fps, self.fail_starts, self.start_delay = fps, 0, 0.0
        self.stall = threading.Event()
        self.sessions = []

    def open_session(self):
        return FakeSession(self)


class FakeZedX(FakeCameraMixin, ZedXCamera):
    def __init__(self, cache_dir, calibration_path=None, **kwargs):
        super().__init__(42757821, Calibration(42757821, str(cache_dir), 'SVGA', calibration_path), **kwargs)
        self.setup_fake()

    def publish_frame(self):
        now_ns = time.time_ns()
        self.pairs.publish((jpeg(50), jpeg(200)), now_ns, time.monotonic_ns())


class FakeNano(FakeCameraMixin, NanoCamera):
    def __init__(self, cache_dir, calibration_path=None, **kwargs):
        super().__init__(['/dev/fake3', '/dev/fake2'], 99292912,
                         Calibration(99292912, str(cache_dir), 'FHD1200', calibration_path), **kwargs)
        self.setup_fake()

    def publish_frame(self):
        # Two free-running sensors: the right one reads out 1 ms after the left one.
        mono, wall = time.monotonic_ns(), time.time_ns()
        self.eye_slots['left'].publish(jpeg(80), wall, mono)
        self.eye_slots['right'].publish(jpeg(160), wall + 1_000_000, mono + 1_000_000)


def wait_for(predicate, timeout=5.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


class Client:
    """Minimal HTTP client for the hub under test (no proxies, one connection per request)."""

    def __init__(self, port, timeout=10.0):
        self.port, self.timeout = port, timeout

    def request(self, method, path, body=None):
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=self.timeout)
        headers = {'Content-Type': 'application/json'} if body is not None else {}
        conn.request(method, path, body=None if body is None else json.dumps(body), headers=headers)
        return conn, conn.getresponse()

    def get(self, path):
        conn, response = self.request('GET', path)
        try:
            return response.status, response.getheader('Content-Type'), response.read(), dict(response.getheaders())
        finally:
            conn.close()

    def json(self, method, path, body=None):
        conn, response = self.request(method, path, body)
        try:
            return response.status, json.loads(response.read())
        finally:
            conn.close()

    def select(self, camera, query=''):
        return self.json('POST', '/select' + query, {'camera': camera})


def read_parts(response):
    """Yield (headers, body) of a multipart/x-mixed-replace response, like the Thor client parses it."""
    while True:
        line = response.readline()
        if not line:
            return
        if not line.startswith(b'--'):
            continue
        headers = {}
        while True:
            line = response.readline()
            if not line:
                return
            if line in (b'\r\n', b'\n'):
                break
            key, _, value = line.decode('latin-1').partition(':')
            headers[key.strip().lower()] = value.strip()
        body = response.read(int(headers['content-length']))
        yield headers, body
