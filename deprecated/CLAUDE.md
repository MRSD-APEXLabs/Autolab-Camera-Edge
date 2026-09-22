# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

```bash
# With uv (recommended)
uv run python main.py

# Or directly if venv is activated
python main.py
```

Requires physical ZED cameras to be connected (detected via `pyzed`). The app exits immediately if no cameras are found.

## Architecture

This is an edge robotics vision system that runs two mutually exclusive camera processing modes, exposed over WebSocket.

**Two WebSocket servers run concurrently:**
- Control server (`port 8765`): receives JSON commands to switch modes and query status
- Stream server (`port 8766`): pushes latest frame payloads to connected clients at `STREAM_HZ`

**Three operating modes** managed by `ModeManager` in `main.py`:
- `servo` — ZED cam 1 (serial `SERIAL_CAM1`) + YOLO OBB model → image-based visual servoing (IBVS) → xArm Cartesian velocity commands
- `inspect` — ZED cam 2 (serial `SERIAL_CAM2`) + YOLO OBB model + AprilTag detection + point cloud acquisition
- `idle` — no camera worker running

**Threading model:**
- Each mode runs as a `CameraWorker` (daemon `threading.Thread`, defined in `cam_worker.py`)
- `ModeManager` holds a lock, stops the current worker, and starts the new one on mode switch
- `StreamHub` (`streamhub.py`) is the thread-safe latest-frame store — workers call `publish(channel, payload)`, the async stream handler calls `snapshot(channel)`

**Data flow:**
```
ZED Camera → CameraWorker.run() → StreamHub.publish()
                                         ↓
WebSocket client ← stream_handler ← StreamHub.snapshot()
```

**Key files:**
- `config.py` — all tunables: camera serials, ARM IP, model paths, ports, IBVS thresholds
- `utils.py` — IBVS math (`image_ibvs_command`, `se3_to_adj`), OBB helpers, xArm gripper commands, JPEG/zlib encoders
- `visual_servo.py` — `ZEDYOLOServo`: opens camera, loads YOLO, connects xArm, runs IBVS loop
- `inspect_cam.py` — `ZEDInspectWorker`: opens camera with depth, runs YOLO + AprilTag + point cloud
- `dummy.py` — standalone prototype (not used by `main.py`); documents the WebSocket message schema

## Configuration

Edit `config.py` before running:
- `SERIAL_CAM1` / `SERIAL_CAM2` — ZED camera serial numbers
- `ARM_IP` — xArm IP address
- `MODEL_SERVO` / `MODEL_INSPECT` — paths to YOLO `.pt` model files (must exist at runtime)
- `DEBUG = True` — skips actual robot motion, prints commands instead

## WebSocket Protocol

**Control messages (port 8765):**
```json
{"cmd": "mode", "mode": "servo|inspect|idle"}
{"cmd": "stop"}
{"cmd": "status"}
```

**Stream payloads (port 8766):**
- Servo: `image_jpeg_b64`, `detections` (OBB list), `target_uv`, timing fields
- Inspect: `image_jpeg_b64`, `pointcloud_zlib_b64` (float32 XYZRGBA, zlib-compressed), `wellplates`, `apriltags`

## Dependencies

Managed with `uv` (`pyproject.toml`). Key non-PyPI dependencies:
- `pyzed` — ZED SDK Python bindings (requires ZED SDK installed separately)
- `xarm` — UFactory xArm SDK
- `ultralytics` — YOLOv8/v11 (not listed in pyproject.toml but imported)
- `apriltag` — AprilTag detector (not listed in pyproject.toml but imported)
- CUDA GPU required — both YOLO models are loaded with `.to("cuda")`
