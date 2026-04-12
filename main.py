#!/usr/bin/env python3
"""
Unified Websocket server

Mode A (servo):
  - ZED camera 1 (wrist)
  - YOLO OBB visual servo
  - xArm control

Mode B (inspect):
  - ZED camera 2 (base)
  - second YOLO model
  - AprilTag detection
  - point cloud acquisition hook

WebSocket servers:
  - Control  (port 8765): mode switching and status
  - Stream   (port 8766): active mode full pipeline feed
  - Raw      (port 8767): raw image feed per camera (subscribe by name)

Only one active worker runs at a time. Both cameras are always open.
A long-lived idle thread grabs raw images from both cameras continuously.
Servo and inspect workers are also long-lived — they wait for start_run()
and return to idle after each run rather than being recreated.
"""

import asyncio
import json
import logging
import sys
import threading
from typing import Optional

import pyzed.sl as sl
import websockets

from image_worker import ZEDImageWorker
from inspect_cam import ZEDInspectWorker
from visual_servo import ZEDYOLOServo
from config import *
from streamhub import StreamHub


# =============================================================================
# Mode manager
# =============================================================================


class ModeManager:
    def __init__(self, arm_ip: str, stream_hub: StreamHub, debug: bool = False):
        self.stream_hub = stream_hub
        self.debug = debug
        self.lock = threading.Lock()
        self.current_mode = "idle"

        # Connect to xArm once for the lifetime of the process
        self._arm = None
        if not debug:
            from xarm.wrapper import XArmAPI
            self._arm = XArmAPI(arm_ip)
            self._arm.connect()
            if not self._arm.connected:
                raise RuntimeError(f"Failed to connect to xArm at {arm_ip}")

        # Open wrist camera (cam1, no depth)
        self._wrist_cam = sl.Camera()
        self._wrist_lock = threading.Lock()
        wrist_init = sl.InitParameters()
        wrist_init.set_from_serial_number(SERIAL_CAM1)
        wrist_init.camera_resolution = sl.RESOLUTION.SVGA
        wrist_init.camera_fps = CAMERA_FPS
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
        base_init.camera_fps = CAMERA_FPS
        base_init.depth_mode = sl.DEPTH_MODE.NEURAL
        base_init.coordinate_units = sl.UNIT.METER
        base_init.depth_minimum_distance = 0.2
        base_init.depth_maximum_distance = 1.7
        err = self._base_cam.open(base_init)
        if err != sl.ERROR_CODE.SUCCESS:
            self._wrist_cam.close()
            raise RuntimeError(f"Base camera {SERIAL_CAM2} failed to open: {err}")

        # Long-lived workers — started once, run for the lifetime of the process
        self._idle_worker = ZEDImageWorker(
            self._wrist_cam, self._wrist_lock,
            self._base_cam, self._base_lock,
            stream_hub=stream_hub,
        )
        self._idle_worker.start()

        self._servo_worker = ZEDYOLOServo(
            self._arm,
            self._wrist_cam, self._wrist_lock,
            stream_hub=stream_hub,
            debug=debug,
        )
        self._servo_worker.start()

        self._inspect_worker = ZEDInspectWorker(
            self._base_cam, self._base_lock,
            stream_hub=stream_hub,
            debug=debug,
        )
        self._inspect_worker.start()

    def _on_servo_complete(self):
        with self.lock:
            self.current_mode = "idle"
            self.stream_hub.set_active_mode("idle")

    def stop(self):
        """Set mode to idle."""
        with self.lock:
            if self.current_mode == "servo":
                self._servo_worker.abort_run()
            elif self.current_mode == "inspect":
                self._inspect_worker.abort_run()
            self.current_mode = "idle"
            self.stream_hub.set_active_mode("idle")

    def switch_mode(self, mode: str):
        mode = mode.lower().strip()
        if mode not in {"idle", "servo", "inspect"}:
            raise ValueError(f"Unsupported mode: {mode}")

        with self.lock:
            if mode == self.current_mode:
                self.stream_hub.set_active_mode(mode)
                return self.current_mode

            # Abort whatever is currently running
            if self.current_mode == "servo":
                self._servo_worker.abort_run()
            elif self.current_mode == "inspect":
                self._inspect_worker.abort_run()

            if mode == "servo":
                self._servo_worker.start_run(on_complete=self._on_servo_complete)
            elif mode == "inspect":
                self._inspect_worker.start_run()

            self.current_mode = mode
            self.stream_hub.set_active_mode(mode)
            return self.current_mode

    def status(self):
        with self.lock:
            return {"mode": self.current_mode, "worker_alive": True}

    def shutdown(self):
        """Stop all workers and close cameras/arm."""
        self._servo_worker.stop()
        self._servo_worker.join(timeout=10.0)

        self._inspect_worker.stop()
        self._inspect_worker.join(timeout=10.0)

        self._idle_worker.stop()
        self._idle_worker.join(timeout=5.0)

        try:
            self._wrist_cam.close()
        except Exception as e:
            print(repr(e))
        try:
            self._base_cam.close()
        except Exception as e:
            print(repr(e))
        try:
            if self._arm is not None:
                self._arm.disconnect()
        except Exception as e:
            print(repr(e))


# =============================================================================
# WebSocket handlers
# =============================================================================

STREAM_SNAPSHOT_INTERVAL = 1.0 / STREAM_HZ
MANAGER: Optional[ModeManager] = None
STREAM_HUB = StreamHub()


async def control_handler(ws, *args):
    await ws.send(json.dumps({"ok": True, "message": "connected"}))

    try:
        async for message in ws:
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                await ws.send(json.dumps({"ok": False, "error": "invalid JSON"}))
                continue

            cmd = str(data.get("cmd", "")).lower()

            try:
                if cmd == "mode":
                    mode = str(data.get("mode", "")).lower()
                    active = MANAGER.switch_mode(mode)
                    await ws.send(json.dumps({"ok": True, "mode": active}))
                elif cmd == "stop":
                    MANAGER.stop()
                    await ws.send(json.dumps({"ok": True, "mode": "idle"}))
                elif cmd == "status":
                    await ws.send(json.dumps({"ok": True, **MANAGER.status()}))
                else:
                    await ws.send(
                        json.dumps({"ok": False, "error": f"unknown cmd: {cmd}"})
                    )
            except Exception as e:
                await ws.send(json.dumps({"ok": False, "error": str(e)}))

    except websockets.exceptions.ConnectionClosed:
        pass


async def stream_handler(ws, *args):
    """Send the latest full-pipeline frame from whichever mode is active."""
    try:
        last_seq = -1
        while True:
            mode = STREAM_HUB.get_active_mode()
            if mode == "servo":
                seq, payload = STREAM_HUB.snapshot("servo")
            elif mode == "inspect":
                seq, payload = STREAM_HUB.snapshot("inspect")
            else:
                seq, payload = -1, None

            if payload is None:
                await ws.send(
                    json.dumps(
                        {
                            "type": "idle",
                            "mode": mode,
                            "message": "waiting for active mode",
                        }
                    )
                )
            elif seq != last_seq:
                out = dict(payload)
                out["mode"] = mode
                await ws.send(json.dumps(out))
                last_seq = seq

            await asyncio.sleep(STREAM_SNAPSHOT_INTERVAL)
    except websockets.exceptions.ConnectionClosed:
        pass


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


# =============================================================================
# Entry point
# =============================================================================

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

    log = logging.getLogger(__name__)
    async with control_server, stream_server, raw_server:
        log.info("Control server:    ws://0.0.0.0:%d", CONTROL_PORT)
        log.info("Stream server:     ws://0.0.0.0:%d", STREAM_PORT)
        log.info("Raw stream server: ws://0.0.0.0:%d", RAW_STREAM_PORT)
        await asyncio.Future()


def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s  %(levelname)-8s  [%(threadName)s]  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    devices = sl.Camera.get_device_list()
    if not devices:
        logging.error("No ZED camera found")
        sys.exit(1)

    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if MANAGER is not None:
                MANAGER.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
