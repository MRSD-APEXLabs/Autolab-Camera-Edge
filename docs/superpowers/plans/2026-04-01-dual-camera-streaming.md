# Dual-Camera Streaming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run both ZED cameras simultaneously — one active thread doing full pipeline processing streamed on port 8766, one idle thread grabbing raw images from both cameras streamed on port 8767.

**Architecture:** Both cameras open at ModeManager startup and stay open for the process lifetime. Each camera has a grab lock shared between the active thread and the idle thread. The idle thread (ZEDImageWorker) runs forever; the active thread (ZEDYOLOServo or ZEDInspectWorker) is swapped on mode changes. Active workers receive injected camera references instead of opening cameras themselves.

**Tech Stack:** Python 3.10, pyzed (ZED SDK), opencv-python, ultralytics (YOLO), websockets, threading, asyncio

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `config.py` | Modify | Add `RAW_STREAM_PORT = 8767` |
| `streamhub.py` | Modify | Add `"wrist_raw"` and `"base_raw"` channels; clean unused imports |
| `cam_worker.py` | Modify | Accept injected `camera` + `grab_lock` instead of `serial`; clean unused imports |
| `visual_servo.py` | Modify | Remove camera open/close; use injected camera + lock; wrap grabs |
| `inspect_cam.py` | Modify | Remove camera open/close; use injected camera + lock; wrap grabs |
| `image_worker.py` | Create | `ZEDImageWorker` — grabs both cameras, publishes raw frames |
| `main.py` | Modify | ModeManager opens cameras + starts idle thread; add `raw_stream_handler`; third WebSocket server; `shutdown()` |
| `tests/test_streamhub.py` | Create | StreamHub channel tests |
| `pyproject.toml` | Modify | Add pytest dev dependency |

---

## Task 1: StreamHub — add raw channels + test infrastructure

**Files:**
- Modify: `streamhub.py`
- Modify: `config.py`
- Modify: `pyproject.toml`
- Create: `tests/__init__.py`
- Create: `tests/test_streamhub.py`

- [ ] **Step 1: Add dev dependencies to pyproject.toml**

```toml
[project]
name = "autolab-camera-edge"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "opencv-python>=4.13.0.92",
    "pyzed>=1.3.0",
    "xarm>=0.0.4",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
]
```

- [ ] **Step 2: Add RAW_STREAM_PORT to config.py**

In `config.py`, add after `STREAM_PORT = 8766`:
```python
RAW_STREAM_PORT = 8767
```

- [ ] **Step 3: Write failing StreamHub tests**

Create `tests/__init__.py` (empty).

Create `tests/test_streamhub.py`:
```python
import pytest
from streamhub import StreamHub


def test_wrist_raw_channel_publish_and_snapshot():
    hub = StreamHub()
    hub.publish("wrist_raw", {"image": "abc"})
    seq, payload = hub.snapshot("wrist_raw")
    assert seq == 1
    assert payload["image"] == "abc"
    assert "frame_id" in payload
    assert "timestamp" in payload


def test_base_raw_channel_publish_and_snapshot():
    hub = StreamHub()
    hub.publish("base_raw", {"image": "xyz"})
    seq, payload = hub.snapshot("base_raw")
    assert seq == 1
    assert payload["image"] == "xyz"


def test_unknown_channel_publish_raises():
    hub = StreamHub()
    with pytest.raises(ValueError, match="Unknown channel"):
        hub.publish("nonexistent", {})


def test_unknown_channel_snapshot_raises():
    hub = StreamHub()
    with pytest.raises(ValueError, match="Unknown channel"):
        hub.snapshot("nonexistent")


def test_channels_are_independent():
    hub = StreamHub()
    hub.publish("wrist_raw", {"data": 1})
    hub.publish("base_raw", {"data": 2})
    _, wrist = hub.snapshot("wrist_raw")
    _, base = hub.snapshot("base_raw")
    assert wrist["data"] == 1
    assert base["data"] == 2
```

- [ ] **Step 4: Run tests — expect ImportError or channel KeyError**

```bash
uv run pytest tests/test_streamhub.py -v
```

Expected: FAIL — `streamhub.py` currently imports pyzed and doesn't have the new channels.

- [ ] **Step 5: Rewrite streamhub.py with clean imports and new channels**

Replace the entire contents of `streamhub.py`:
```python
import threading
import time
from typing import Any, Dict, Optional, Tuple


class StreamHub:
    """Thread-safe latest-frame store for all streaming channels."""

    _CHANNELS = {"servo", "inspect", "wrist_raw", "base_raw"}

    def __init__(self):
        self._lock = threading.Lock()
        self._seq: Dict[str, int] = {ch: 0 for ch in self._CHANNELS}
        self._latest: Dict[str, Optional[Dict[str, Any]]] = {ch: None for ch in self._CHANNELS}
        self._active_mode = "idle"

    def set_active_mode(self, mode: str):
        with self._lock:
            self._active_mode = mode

    def get_active_mode(self) -> str:
        with self._lock:
            return self._active_mode

    def publish(self, channel: str, payload: Dict[str, Any]):
        if channel not in self._CHANNELS:
            raise ValueError(f"Unknown channel: {channel}")
        with self._lock:
            self._seq[channel] += 1
            payload = dict(payload)
            payload["frame_id"] = self._seq[channel]
            payload["timestamp"] = time.time()
            self._latest[channel] = payload

    def snapshot(self, channel: str) -> Tuple[int, Optional[Dict[str, Any]]]:
        if channel not in self._CHANNELS:
            raise ValueError(f"Unknown channel: {channel}")
        with self._lock:
            if self._latest[channel] is None:
                return self._seq[channel], None
            return self._seq[channel], dict(self._latest[channel])
```

- [ ] **Step 6: Run tests — expect all pass**

```bash
uv run pytest tests/test_streamhub.py -v
```

Expected: 5 passed.

---

## Task 2: Refactor CameraWorker base class

**Files:**
- Modify: `cam_worker.py`

- [ ] **Step 1: Rewrite cam_worker.py with clean imports and new signature**

Replace the entire contents of `cam_worker.py`:
```python
import threading
from typing import Optional


class CameraWorker(threading.Thread):
    """Base class for single-camera processing threads.

    Subclasses receive an already-open camera and a grab lock.
    They must NOT open or close the camera.
    All camera.grab() calls must be made while holding grab_lock.
    """

    def __init__(self, name: str, camera, grab_lock: threading.Lock):
        super().__init__(daemon=True)
        self.name = name
        self.camera = camera
        self.grab_lock = grab_lock
        self.stop_event = threading.Event()
        self.exc: Optional[BaseException] = None

    def stop(self):
        self.stop_event.set()

    def should_stop(self) -> bool:
        return self.stop_event.is_set()

    def run(self):
        raise NotImplementedError
```

- [ ] **Step 2: Verify no import errors**

```bash
uv run python -c "from cam_worker import CameraWorker; print('OK')"
```

Expected: `OK`

---

## Task 3: Refactor ZEDYOLOServo

**Files:**
- Modify: `visual_servo.py`

The changes:
- Remove `self.camera = sl.Camera()`, `sl.InitParameters` block, and `self.camera.open()` from `__init__` (camera is injected via `super().__init__`)
- Change signature to accept `camera` and `grab_lock` instead of `serial`
- Remove `self.camera.close()` from `cleanup()` (ModeManager owns camera lifetime)
- Wrap `camera.grab()` + `camera.retrieve_image()` together inside `with self.grab_lock` in `run()`

- [ ] **Step 1: Update __init__ signature and remove camera-open logic**

In `visual_servo.py`, replace the `ZEDYOLOServo.__init__` method:

```python
def __init__(
    self, ip: str, camera, grab_lock: threading.Lock, stream_hub: StreamHub, debug: bool = False
):
    super().__init__("servo", camera, grab_lock)
    self.ip = ip
    self.debug = debug
    self.stream_hub = stream_hub
    self.runtime = sl.RuntimeParameters()
    self.view = sl.VIEW.RIGHT
    self.model = YOLO(MODEL_SERVO)
    self.model.to("cuda")
    self.threshold_action_done = False

    self.T_g_cr = np.array(
        [
            [1.000, 0.000, 0.000, 0.0],
            [0.000, 1.000, 0.000, 0.0],
            [0.000, 0.000, 1.000, 0.0],
            [0.000, 0.000, 0.000, 1.0],
        ],
        dtype=np.float64,
    )
    self.Ad_g_c = se3_to_adj(self.T_g_cr)

    cam_info = self.camera.get_camera_information()
    right_calib = cam_info.camera_configuration.calibration_parameters.right_cam
    self.fx = float(right_calib.fx)
    self.fy = float(right_calib.fy)
    self.cx = float(right_calib.cx)
    self.cy = float(right_calib.cy)

    dummy = np.zeros((600, 800, 3), dtype=np.uint8)
    for _ in range(3):
        _ = self.model(dummy, verbose=False)

    self.arm = XArmAPI(ip)
    self.arm.connect()
    if not self.arm.connected:
        raise RuntimeError(f"Failed to connect to xArm at {ip}")

    self.arm.motion_enable(True)
    self.arm.set_mode(0)
    self.arm.set_state(0)

    setup_gripper(self.arm)
    gripper_open(self.arm)
```

- [ ] **Step 2: Remove camera.close() from cleanup()**

In `visual_servo.py`, replace the `cleanup` method:
```python
def cleanup(self):
    try:
        self.stop_robot()
    except Exception:
        pass

    try:
        if not self.debug:
            self.arm.set_mode(0)
            self.arm.set_state(0)
    except Exception:
        pass

    try:
        cv2.destroyWindow("visual_servoing_right")
    except Exception:
        pass
```

- [ ] **Step 3: Wrap grab + retrieve_image with grab_lock in run()**

In `visual_servo.py`, replace the grab section at the top of the `while not self.should_stop()` loop. The current code is:
```python
err = self.camera.grab(self.runtime)
if err != sl.ERROR_CODE.SUCCESS:
    time.sleep(0.001)
    continue

t0 = time.time()
right_frame = sl.Mat()
self.camera.retrieve_image(right_frame, self.view)
```

Replace with:
```python
right_frame = sl.Mat()
with self.grab_lock:
    err = self.camera.grab(self.runtime)
    if err == sl.ERROR_CODE.SUCCESS:
        self.camera.retrieve_image(right_frame, self.view)

if err != sl.ERROR_CODE.SUCCESS:
    time.sleep(0.001)
    continue

t0 = time.time()
```

- [ ] **Step 4: Verify no syntax errors**

```bash
uv run python -c "from visual_servo import ZEDYOLOServo; print('OK')"
```

Expected: `OK` (or ImportError only from missing hardware libraries, not syntax errors)

---

## Task 4: Refactor ZEDInspectWorker

**Files:**
- Modify: `inspect_cam.py`

Same pattern as Task 3: remove camera-open logic, accept injected camera + lock, remove camera.close(), wrap grab + retrieves.

- [ ] **Step 1: Update __init__ signature and remove camera-open logic**

In `inspect_cam.py`, replace the `ZEDInspectWorker.__init__` method:

```python
def __init__(self, camera, grab_lock: threading.Lock, stream_hub: StreamHub, debug: bool = False):
    super().__init__("inspect", camera, grab_lock)
    self.debug = debug
    self.stream_hub = stream_hub
    self.runtime = sl.RuntimeParameters()
    self.frame_mat = sl.Mat()
    self.pc_mat = sl.Mat()
    self.model = YOLO(MODEL_INSPECT)
    self.model.to("cuda")
    self.apriltag_detector = apriltag.Detector(
        apriltag.DetectorOptions(families="tag16h5")
    )

    dummy = np.zeros((600, 800, 3), dtype=np.uint8)
    for _ in range(3):
        _ = self.model(dummy, verbose=False)
```

- [ ] **Step 2: Remove camera.close() from cleanup()**

In `inspect_cam.py`, replace the `cleanup` method:
```python
def cleanup(self):
    try:
        cv2.destroyWindow("inspect_camera")
    except Exception:
        pass
```

- [ ] **Step 3: Wrap grab + retrieve calls with grab_lock in run()**

In `inspect_cam.py`, replace the grab section at the top of the `while not self.should_stop()` loop. The current code is:
```python
start = time.time()
err = self.camera.grab(self.runtime)
if err != sl.ERROR_CODE.SUCCESS:
    time.sleep(0.001)
    continue

self.camera.retrieve_image(self.frame_mat, sl.VIEW.LEFT)
```

Replace with:
```python
start = time.time()
with self.grab_lock:
    err = self.camera.grab(self.runtime)
    if err == sl.ERROR_CODE.SUCCESS:
        self.camera.retrieve_image(self.frame_mat, sl.VIEW.LEFT)

if err != sl.ERROR_CODE.SUCCESS:
    time.sleep(0.001)
    continue
```

Then find the point cloud retrieval — currently:
```python
self.camera.retrieve_measure(self.pc_mat, sl.MEASURE.XYZRGBA)
```

This must also be inside the grab lock to avoid reading a stale buffer after the idle thread grabs. Since it's now separated from the grab, wrap it too. Replace the entire grab+retrieve block to include point cloud:

```python
start = time.time()
with self.grab_lock:
    err = self.camera.grab(self.runtime)
    if err == sl.ERROR_CODE.SUCCESS:
        self.camera.retrieve_image(self.frame_mat, sl.VIEW.LEFT)
        self.camera.retrieve_measure(self.pc_mat, sl.MEASURE.XYZRGBA)

if err != sl.ERROR_CODE.SUCCESS:
    time.sleep(0.001)
    continue
```

And remove the standalone `self.camera.retrieve_measure(self.pc_mat, sl.MEASURE.XYZRGBA)` line that appears later in the loop.

- [ ] **Step 4: Verify no syntax errors**

```bash
uv run python -c "from inspect_cam import ZEDInspectWorker; print('OK')"
```

Expected: `OK`

---

## Task 5: Create ZEDImageWorker

**Files:**
- Create: `image_worker.py`

- [ ] **Step 1: Create image_worker.py**

```python
import threading
import time

import cv2
import pyzed.sl as sl

from streamhub import StreamHub
from utils import encode_jpeg_b64


class ZEDImageWorker(threading.Thread):
    """Long-lived thread that grabs raw images from both cameras and publishes them.

    Serializes each camera's grab + retrieve_image within the shared grab lock.
    Never opens or closes cameras — lifetime is managed by ModeManager.
    """

    def __init__(
        self,
        wrist_cam: sl.Camera,
        wrist_lock: threading.Lock,
        base_cam: sl.Camera,
        base_lock: threading.Lock,
        stream_hub: StreamHub,
    ):
        super().__init__(daemon=True)
        self.wrist_cam = wrist_cam
        self.wrist_lock = wrist_lock
        self.base_cam = base_cam
        self.base_lock = base_lock
        self.stream_hub = stream_hub
        self.stop_event = threading.Event()
        self.wrist_runtime = sl.RuntimeParameters()
        self.base_runtime = sl.RuntimeParameters()
        self.wrist_mat = sl.Mat()
        self.base_mat = sl.Mat()

    def stop(self):
        self.stop_event.set()

    def run(self):
        while not self.stop_event.is_set():
            wrist_ok = False
            with self.wrist_lock:
                err = self.wrist_cam.grab(self.wrist_runtime)
                if err == sl.ERROR_CODE.SUCCESS:
                    self.wrist_cam.retrieve_image(self.wrist_mat, sl.VIEW.RIGHT)
                    wrist_ok = True

            if wrist_ok:
                frame = cv2.cvtColor(self.wrist_mat.get_data(), cv2.COLOR_BGRA2BGR)
                self.stream_hub.publish(
                    "wrist_raw",
                    {
                        "type": "raw_frame",
                        "camera": "wrist",
                        "image_jpeg_b64": encode_jpeg_b64(frame),
                    },
                )

            base_ok = False
            with self.base_lock:
                err = self.base_cam.grab(self.base_runtime)
                if err == sl.ERROR_CODE.SUCCESS:
                    self.base_cam.retrieve_image(self.base_mat, sl.VIEW.LEFT)
                    base_ok = True

            if base_ok:
                frame = cv2.cvtColor(self.base_mat.get_data(), cv2.COLOR_BGRA2BGR)
                self.stream_hub.publish(
                    "base_raw",
                    {
                        "type": "raw_frame",
                        "camera": "base",
                        "image_jpeg_b64": encode_jpeg_b64(frame),
                    },
                )

            if not wrist_ok and not base_ok:
                time.sleep(0.001)
```

- [ ] **Step 2: Verify no syntax errors**

```bash
uv run python -c "from image_worker import ZEDImageWorker; print('OK')"
```

Expected: `OK`

---

## Task 6: Refactor ModeManager

**Files:**
- Modify: `main.py`

ModeManager gains: camera open at startup, grab locks, idle thread creation, updated `switch_mode`, new `shutdown()`. The `stop()` method (sets to idle) is unchanged.

- [ ] **Step 1: Add import for ZEDImageWorker at top of main.py**

In `main.py`, after the existing imports, add:
```python
from image_worker import ZEDImageWorker
```

- [ ] **Step 2: Replace ModeManager.__init__**

```python
def __init__(self, arm_ip: str, stream_hub: StreamHub, debug: bool = False):
    self.arm_ip = arm_ip
    self.stream_hub = stream_hub
    self.debug = debug
    self.lock = threading.Lock()
    self.current_mode = "idle"
    self.worker: Optional[CameraWorker] = None

    # Open wrist camera (cam1, no depth)
    self._wrist_cam = sl.Camera()
    self._wrist_lock = threading.Lock()
    wrist_init = sl.InitParameters()
    wrist_init.set_from_serial_number(SERIAL_CAM1)
    wrist_init.camera_resolution = sl.RESOLUTION.SVGA
    wrist_init.camera_fps = 30
    wrist_init.depth_mode = sl.DEPTH_MODE.NONE
    err = self._wrist_cam.open(wrist_init)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Wrist camera {SERIAL_CAM1} failed to open: {err}")

    # Open base camera (cam2, NEURAL depth)
    self._base_cam = sl.Camera()
    self._base_lock = threading.Lock()
    base_init = sl.InitParameters()
    base_init.set_from_serial_number(SERIAL_CAM2)
    base_init.camera_resolution = sl.RESOLUTION.SVGA
    base_init.camera_fps = 30
    base_init.depth_mode = sl.DEPTH_MODE.NEURAL
    base_init.coordinate_units = sl.UNIT.METER
    base_init.depth_minimum_distance = 0.2
    base_init.depth_maximum_distance = 1.7
    err = self._base_cam.open(base_init)
    if err != sl.ERROR_CODE.SUCCESS:
        self._wrist_cam.close()
        raise RuntimeError(f"Base camera {SERIAL_CAM2} failed to open: {err}")

    # Start idle thread — runs for the lifetime of the process
    self._idle_worker = ZEDImageWorker(
        self._wrist_cam, self._wrist_lock,
        self._base_cam, self._base_lock,
        stream_hub=stream_hub,
    )
    self._idle_worker.start()
```

- [ ] **Step 3: Replace ModeManager.switch_mode**

```python
def switch_mode(self, mode: str):
    mode = mode.lower().strip()
    if mode not in {"idle", "servo", "inspect"}:
        raise ValueError(f"Unsupported mode: {mode}")

    with self.lock:
        if mode == self.current_mode:
            self.stream_hub.set_active_mode(mode)
            return self.current_mode

        self._stop_worker_locked()

        if mode == "servo":
            self.worker = ZEDYOLOServo(
                self.arm_ip,
                self._wrist_cam,
                self._wrist_lock,
                stream_hub=self.stream_hub,
                debug=self.debug,
            )
            self.worker.start()
        elif mode == "inspect":
            self.worker = ZEDInspectWorker(
                self._base_cam,
                self._base_lock,
                stream_hub=self.stream_hub,
                debug=self.debug,
            )
            self.worker.start()
        else:
            self.worker = None

        self.current_mode = mode
        self.stream_hub.set_active_mode(mode)
        return self.current_mode
```

- [ ] **Step 4: Add ModeManager.shutdown()**

Add this method after `stop()`:
```python
def shutdown(self):
    """Stop active worker, stop idle thread, close both cameras."""
    with self.lock:
        self._stop_worker_locked()
        self.current_mode = "idle"

    self._idle_worker.stop()
    self._idle_worker.join(timeout=5.0)

    try:
        self._wrist_cam.close()
    except Exception:
        pass
    try:
        self._base_cam.close()
    except Exception:
        pass
```

---

## Task 7: Add raw stream handler and wire everything in main()

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Add raw_stream_handler function**

Add this after `stream_handler` in `main.py`:

```python
async def raw_stream_handler(ws, *args):
    """Subscribe to a single camera's raw image feed.

    Client sends: {"camera": "wrist"} or {"camera": "base"}
    Server streams raw JPEG frames at STREAM_HZ.
    """
    try:
        message = await asyncio.wait_for(ws.recv(), timeout=5.0)
        data = json.loads(message)
    except (asyncio.TimeoutError, json.JSONDecodeError, Exception):
        await ws.send(json.dumps({"ok": False, "error": 'expected {"camera": "wrist"|"base"}'}))
        return

    camera = str(data.get("camera", "")).lower()
    if camera not in {"wrist", "base"}:
        await ws.send(json.dumps({"ok": False, "error": "camera must be 'wrist' or 'base'"}))
        return

    channel = f"{camera}_raw"
    last_seq = -1

    try:
        while True:
            seq, payload = STREAM_HUB.snapshot(channel)
            if payload is not None and seq != last_seq:
                await ws.send(json.dumps(payload))
                last_seq = seq
            await asyncio.sleep(STREAM_SNAPSHOT_INTERVAL)
    except websockets.exceptions.ConnectionClosed:
        pass
```

- [ ] **Step 2: Update main_async() to start third server**

Replace `main_async`:
```python
async def main_async():
    global MANAGER
    MANAGER = ModeManager(ARM_IP, stream_hub=STREAM_HUB, debug=DEBUG)

    control_server = websockets.serve(
        control_handler, "0.0.0.0", CONTROL_PORT, max_size=None
    )
    stream_server = websockets.serve(
        stream_handler, "0.0.0.0", STREAM_PORT, max_size=None
    )
    raw_server = websockets.serve(
        raw_stream_handler, "0.0.0.0", RAW_STREAM_PORT, max_size=None
    )

    async with control_server, stream_server, raw_server:
        print(f"Control server:    ws://0.0.0.0:{CONTROL_PORT}")
        print(f"Stream server:     ws://0.0.0.0:{STREAM_PORT}")
        print(f"Raw stream server: ws://0.0.0.0:{RAW_STREAM_PORT}")
        print(
            'Commands: {"cmd":"mode","mode":"servo"|"inspect"|"idle"}, {"cmd":"status"}, {"cmd":"stop"}'
        )
        await asyncio.Future()
```

- [ ] **Step 3: Update main() to call shutdown() instead of stop()**

Replace the `finally` block in `main()`:
```python
finally:
    try:
        if MANAGER is not None:
            MANAGER.shutdown()
    except Exception:
        pass
```

- [ ] **Step 4: Verify no syntax errors in main.py**

```bash
uv run python -c "import ast; ast.parse(open('main.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Run StreamHub tests to confirm nothing regressed**

```bash
uv run pytest tests/test_streamhub.py -v
```

Expected: 5 passed.
