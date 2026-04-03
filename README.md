# Autolab Camera Edge

Edge robotics vision system that runs two ZED stereo cameras simultaneously, exposed over WebSocket. Designed to run on a GPU-equipped edge computer connected to a UFactory xArm robot.

## Hardware Requirements

- 2× ZED stereo cameras (serials configured in `config.py`)
- UFactory xArm robot
- CUDA-capable GPU (YOLO models run on CUDA)
- ZED SDK installed separately (provides `pyzed`)

## Setup

```bash
# Install dependencies (requires ZED SDK + xArm SDK pre-installed)
uv sync

# Edit config before running
# Set SERIAL_CAM1, SERIAL_CAM2, ARM_IP, and model paths
vim config.py

# Run
uv run python main.py
```

## Architecture

Three WebSocket servers run concurrently:

| Port | Purpose |
|------|---------|
| 8765 | Control — send mode commands, query status |
| 8766 | Active feed — full pipeline output from the current mode |
| 8767 | Raw feed — JPEG images from either camera (subscribe by name) |

**Two threads always run:**

- **Active thread** — runs the full pipeline for the current mode (YOLO + IBVS + xArm for `servo`; YOLO + AprilTag + point cloud for `inspect`). Replaced on mode switch.
- **Idle thread** (`ZEDImageWorker`) — grabs raw images from both cameras continuously and publishes them to port 8767. Never stops.

Both cameras are opened once at startup and stay open. Grab operations are serialized per-camera with a `threading.Lock` shared between the active thread and the idle thread.

## WebSocket Protocol

### Control (port 8765)

```json
{"cmd": "mode", "mode": "servo"}
{"cmd": "mode", "mode": "inspect"}
{"cmd": "mode", "mode": "idle"}
{"cmd": "stop"}
{"cmd": "status"}
```

### Active feed (port 8766)

Receives frames automatically. Payload varies by mode:

**Servo:**
```json
{
  "type": "servo_frame",
  "mode": "servo",
  "image_jpeg_b64": "...",
  "detections": [{"center": [u, v], "size": [w, h], "theta": 0.0, "conf": 0.95, "cls_id": 0}],
  "target_uv": [u, v],
  "fps": 29.7,
  "capture_ms": 5.1,
  "inference_ms": 12.3,
  "post_ms": 2.1,
  "frame_id": 42,
  "timestamp": 1234567890.123
}
```

**Inspect:**
```json
{
  "type": "inspect_frame",
  "mode": "inspect",
  "image_jpeg_b64": "...",
  "pointcloud_zlib_b64": "...",
  "pointcloud_shape": [N, 4],
  "pointcloud_dtype": "float32",
  "wellplates": [...],
  "apriltags": [{"id": 0, "center": [u, v], "corners": [[u,v], ...]}],
  "pc_count": 12345,
  "fps": 14.2,
  "frame_id": 42,
  "timestamp": 1234567890.123
}
```

Point cloud data is float32 XYZRGBA, zlib-compressed, base64-encoded. Decode with:
```python
import zlib, base64, numpy as np
raw = zlib.decompress(base64.b64decode(msg["pointcloud_zlib_b64"]))
pc = np.frombuffer(raw, dtype=np.float32).reshape(msg["pointcloud_shape"])
```

### Raw feed (port 8767)

On connect, send a subscribe message:
```json
{"camera": "wrist"}
```
or
```json
{"camera": "base"}
```

Then receive a stream of frames:
```json
{
  "type": "raw_frame",
  "camera": "wrist",
  "image_jpeg_b64": "...",
  "frame_id": 42,
  "timestamp": 1234567890.123
}
```

Invalid subscribe messages receive `{"ok": false, "error": "..."}` and the connection closes.

## Configuration (`config.py`)

| Variable | Description |
|---|---|
| `SERIAL_CAM1` | ZED serial for wrist camera (used in `servo` mode) |
| `SERIAL_CAM2` | ZED serial for base camera (used in `inspect` mode) |
| `ARM_IP` | xArm IP address |
| `MODEL_SERVO` | Path to YOLO `.pt` model for servo mode |
| `MODEL_INSPECT` | Path to YOLO `.pt` model for inspect mode |
| `DEBUG` | If `True`, skips robot motion and prints commands instead |
| `STREAM_HZ` | Frame rate for WebSocket stream output |
| `POINTCLOUD_STRIDE` | Subsample factor for point cloud (larger = smaller payload) |
| `AREA_STOP_THRESHOLD` | OBB area (px²) that triggers the grasp sequence in servo mode |

## Running Tests

```bash
uv run pytest tests/ -v
```

Tests require no hardware — `StreamHub` tests run with stdlib only.
