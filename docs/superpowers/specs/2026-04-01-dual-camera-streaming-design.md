# Dual-Camera Streaming Design

**Date:** 2026-04-01  
**Status:** Approved

## Problem

The system currently allows only one camera to be active at a time. Switching modes (servo ↔ inspect) stops the current camera worker and starts the new one. No raw image stream exists for the inactive camera, and both streams share port 8766, making independent consumption impossible.

## Goals

1. Command whether the system is in `servo` or `inspect` mode (unchanged semantics).
2. Stream the active mode's full pipeline output (image + detections/pointcloud/robot data) on port 8766.
3. Stream raw images from both cameras simultaneously on port 8767; clients subscribe to a specific camera (`wrist` or `base`).

## Architecture

### Two Concurrent Threads

**Active thread** — one at a time, restarts on mode switch:
- Holds a reference to one camera + that camera's grab lock.
- Runs the full pipeline for the current mode (IBVS + xArm for servo; YOLO + AprilTag + point cloud for inspect).
- Publishes full payload to `StreamHub` channel `"servo"` or `"inspect"`.

**Idle thread** — starts at process startup, never stops:
- Holds references to both cameras + both grab locks.
- Loop: acquire wrist lock → grab → release → JPEG encode → publish `"wrist_raw"`; acquire base lock → grab → release → JPEG encode → publish `"base_raw"`.
- No YOLO, no robot logic.
- Unaffected by mode switches.

### Camera Lifecycle

Both `sl.Camera` objects are opened once at `ModeManager` startup and remain open for the lifetime of the process. Each camera has a `threading.Lock` for grab serialization. Workers receive camera references and locks — they do not open or close cameras.

The wrist camera's grab lock is shared between the active thread (in servo mode) and the idle thread. The base camera's grab lock is shared between the active thread (in inspect mode) and the idle thread. Only one thread holds a grab lock at a time.

### Mode Switch

On switch: stop the active thread only. Start the new active thread with the appropriate camera reference and grab lock. The idle thread continues uninterrupted.

On idle: stop the active thread. No active thread runs; the idle thread continues.

### StreamHub

Four channels:

| Channel | Publisher | Description |
|---|---|---|
| `"servo"` | Active thread (servo mode) | Full IBVS payload |
| `"inspect"` | Active thread (inspect mode) | Full inspect payload |
| `"wrist_raw"` | Idle thread | Raw JPEG from wrist camera |
| `"base_raw"` | Idle thread | Raw JPEG from base camera |

### WebSocket Servers

| Port | Purpose | Protocol |
|---|---|---|
| 8765 | Control | Existing command protocol (unchanged) |
| 8766 | Active feed | Existing stream protocol (unchanged) |
| 8767 | Raw dual-camera feed | Subscribe then stream (see below) |

**Port 8767 protocol:**

On connect, client sends:
```json
{"camera": "wrist"}
```
or
```json
{"camera": "base"}
```

Server responds with a stream of frames at `STREAM_HZ`:
```json
{
  "type": "raw_frame",
  "camera": "wrist",
  "frame_id": 42,
  "timestamp": 1234567890.123,
  "image_jpeg_b64": "..."
}
```

Invalid subscribe messages receive an error response and the connection is closed.

## Changes Required

### `config.py`
- Add `RAW_STREAM_PORT = 8767`

### `cam_worker.py`
- `CameraWorker.__init__` accepts `camera: sl.Camera` and `grab_lock: threading.Lock` instead of `serial: str`
- Subclasses use `self.grab_lock` around every `self.camera.grab()` call

### New: `image_worker.py` (or `idle_worker.py`)
- `ZEDImageWorker(threading.Thread)` — not a `CameraWorker` subclass since it holds two cameras
- `__init__(wrist_cam, wrist_lock, base_cam, base_lock, stream_hub)`
- `run()`: grab loop for both cameras with respective locks, publish to `"wrist_raw"` and `"base_raw"`

### `streamhub.py`
- Add `"wrist_raw"` and `"base_raw"` to the channel registry

### `visual_servo.py`
- Remove camera open logic; use injected camera + lock
- Wrap `camera.grab()` with `with self.grab_lock`

### `inspect_cam.py`
- Same changes as `visual_servo.py`

### `main.py`
- `ModeManager.__init__`: open both cameras, create both locks, instantiate and start `ZEDImageWorker`
- `ModeManager.switch_mode`: pass camera ref + lock to active worker constructor
- `ModeManager.stop`: stop active thread only; idle thread is not managed here
- Add `raw_stream_handler` for port 8767
- Start third WebSocket server on `RAW_STREAM_PORT`

## Data Flow

```
Wrist Camera ──grab_lock_wrist──┬── Active thread (servo) ──→ "servo" → port 8766
                                └── Idle thread ────────────→ "wrist_raw" ─┐
                                                                             ├─→ port 8767
Base Camera ───grab_lock_base───┬── Active thread (inspect) → "inspect" → port 8766
                                └── Idle thread ────────────→ "base_raw" ──┘
```
