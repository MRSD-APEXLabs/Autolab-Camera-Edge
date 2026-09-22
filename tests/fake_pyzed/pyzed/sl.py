"""Stand-in for pyzed.sl (the ZED SDK): 60 fps grabs, BGRA images whose value tells the eye apart.

Tests use it in-process by setting the Camera class attributes, and in zedx_capture.py processes
by putting tests/fake_pyzed first on PYTHONPATH: FAKE_ZED (JSON with the same attribute names)
configures it there, and FAKE_ZED_LOG names a file that records open/close across processes.
"""
import atexit
import ctypes
import enum
import json
import os
import time
import types

import numpy as np

CONFIG = json.loads(os.environ.get('FAKE_ZED', '{}'))
if CONFIG.pop('missing', False):
    raise ImportError('fake ZED SDK configured as missing')


class ERROR_CODE(enum.Enum):
    SUCCESS = 0
    CAMERA_NOT_DETECTED = 1
    CAMERA_REBOOTING = 2


RESOLUTION = types.SimpleNamespace(SVGA='SVGA')
DEPTH_MODE = types.SimpleNamespace(NONE='NONE')
VIEW = types.SimpleNamespace(LEFT_UNRECTIFIED='left', RIGHT_UNRECTIFIED='right')
TIME_REFERENCE = types.SimpleNamespace(IMAGE='image')
RuntimeParameters = types.SimpleNamespace


class InitParameters(types.SimpleNamespace):
    def set_from_serial_number(self, serial):
        self.serial = serial


class Mat:
    def get_data(self):
        return self.data


def hold_gil(seconds):
    """Sleep without releasing the GIL, as pyzed's open() and close() do (PyDLL keeps the GIL)."""
    if seconds > 0:
        ctypes.PyDLL(None).usleep(int(seconds * 1e6))


def _log(event):
    path = os.environ.get('FAKE_ZED_LOG')
    if path:
        with open(path, 'a') as f:
            f.write('%s %d\n' % (event, os.getpid()))


def _opens_logged(path):
    try:
        with open(path) as f:
            return sum(1 for line in f if line.startswith('open '))
    except FileNotFoundError:
        return 0


class Camera:
    instances = []
    open_results = []          # ERROR_CODE names of successive open() calls (across processes with a log)
    open_hold_s = 0.0          # open() holds the GIL this long
    resolution = (960, 600)
    failing = None             # (start, end) monotonic window in which grab() fails
    fail_after = None          # grab() fails for good after this many grabs
    hang_after = None          # grab() blocks for good after this many grabs
    crash_after = None         # the process dies (exit code 3) after this many grabs
    noise = False              # open() prints to stdout, as SDK log lines would
    exit_hold_s = 0.0          # the process takes this long to exit, like the SDK's teardown
    clock_offset_ns = 0        # added to the image stamps

    def __init__(self):
        self.init = None
        self.grabs, self.retrieved, self.closed = 0, [], False
        self.next_frame = time.monotonic()
        self.stamp_ns = 0
        Camera.instances.append(self)

    def open(self, init):
        self.init = init
        path = os.environ.get('FAKE_ZED_LOG')
        index = _opens_logged(path) if path else len(Camera.instances) - 1
        _log('open')
        if Camera.noise:
            print('[ZED][fake] opening camera')
            os.write(1, b'[ZED][fake] C-level log line\n')
        hold_gil(Camera.open_hold_s)
        self.next_frame = time.monotonic()
        results = Camera.open_results
        return ERROR_CODE[results[index]] if index < len(results) else ERROR_CODE.SUCCESS

    def get_camera_information(self):
        width, height = Camera.resolution
        resolution = types.SimpleNamespace(width=width, height=height)
        return types.SimpleNamespace(camera_configuration=types.SimpleNamespace(resolution=resolution))

    def grab(self, runtime):
        self.runtime = runtime
        if Camera.hang_after is not None and self.grabs >= Camera.hang_after:
            _log('hang')
            time.sleep(3600)   # a wedged SDK call: SIGTERM's handler runs, but nothing checks it
        if Camera.crash_after is not None and self.grabs >= Camera.crash_after:
            os._exit(3)
        self.next_frame += 1 / 60
        time.sleep(max(0.0, self.next_frame - time.monotonic()))
        window = Camera.failing
        if (window and window[0] <= time.monotonic() < window[1]) or \
                (Camera.fail_after is not None and self.grabs >= Camera.fail_after):
            return ERROR_CODE.CAMERA_REBOOTING
        self.grabs += 1
        # The SDK stamps the exposure, ~30 ms before grab() returns.
        self.stamp_ns = time.time_ns() - 30_000_000 + Camera.clock_offset_ns
        return ERROR_CODE.SUCCESS

    def get_timestamp(self, reference):
        assert reference == 'image'
        return types.SimpleNamespace(get_nanoseconds=lambda: self.stamp_ns)

    def retrieve_image(self, mat, view):
        self.retrieved.append(view)
        mat.data = np.full((600, 960, 4), 60 if view == 'left' else 190, np.uint8)
        return ERROR_CODE.SUCCESS

    def close(self):
        self.closed = True
        _log('close')


for _key, _value in CONFIG.items():
    setattr(Camera, _key, _value)
if Camera.exit_hold_s:
    atexit.register(time.sleep, Camera.exit_hold_s)
