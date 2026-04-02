#!/usr/bin/env python3
"""
Unified Websocket server

Mode A (servo):
  - ZED camera 1
  - YOLO OBB visual servo
  - xArm control

Mode B (inspect):
  - ZED camera 2
  - second YOLO model
  - AprilTag detection
  - point cloud acquisition hook

A websocket control server receives commands such as:
  {"cmd": "mode", "mode": "servo"}
  {"cmd": "mode", "mode": "inspect"}
  {"cmd": "mode", "mode": "idle"}
  {"cmd": "stop"}

Only one camera worker is active at a time. Switching modes stops the
current worker, closes the camera, and starts the new one.

This file keeps your existing processing logic intact and adds the
resource-management layer needed to combine both scripts cleanly.
"""

import asyncio
import json
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import pyzed.sl as sl
import websockets
from ultralytics import YOLO
from xarm.wrapper import XArmAPI
import apriltag
from utils import *
from inspect_cam import ZEDInspectWorker
from visual_servo import ZEDYOLOServo
from config import *
from streamhub import StreamHub

# =============================================================================
# Mode manager
# =============================================================================

class ModeManager:
    def __init__(self, arm_ip: str, stream_hub: StreamHub, debug: bool = False):
        self.arm_ip = arm_ip
        self.stream_hub = stream_hub
        self.debug = debug
        self.lock = threading.Lock()
        self.current_mode = "idle"
        self.worker: Optional[CameraWorker] = None

    def _stop_worker_locked(self):
        if self.worker is None:
            return
        self.worker.stop()
        self.worker.join(timeout=10.0)
        self.worker = None

    def stop(self):
        with self.lock:
            self._stop_worker_locked()
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

            self._stop_worker_locked()

            if mode == "servo":
                self.worker = ZEDYOLOServo(self.arm_ip, SERIAL_CAM1, stream_hub=self.stream_hub, debug=self.debug)
                self.worker.start()
            elif mode == "inspect":
                self.worker = ZEDInspectWorker(SERIAL_CAM2, stream_hub=self.stream_hub, debug=self.debug)
                self.worker.start()
            else:
                self.worker = None

            self.current_mode = mode
            self.stream_hub.set_active_mode(mode)
            return self.current_mode

    def status(self):
        with self.lock:
            alive = bool(self.worker and self.worker.is_alive())
            return {"mode": self.current_mode, "worker_alive": alive}


# =============================================================================
# Websocket control server
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
                    await ws.send(json.dumps({"ok": False, "error": f"unknown cmd: {cmd}"}))
            except Exception as e:
                await ws.send(json.dumps({"ok": False, "error": str(e)}))

    except websockets.exceptions.ConnectionClosed:
        pass


async def stream_handler(ws, *args):
    """Send the latest frame from whichever mode is active."""
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
                await ws.send(json.dumps({"type": "idle", "mode": mode, "message": "waiting for active mode"}))
            elif seq != last_seq:
                out = dict(payload)
                out["mode"] = mode
                await ws.send(json.dumps(out))
                last_seq = seq

            await asyncio.sleep(STREAM_SNAPSHOT_INTERVAL)
    except websockets.exceptions.ConnectionClosed:
        pass


async def main_async():
    global MANAGER
    MANAGER = ModeManager(ARM_IP, stream_hub=STREAM_HUB, debug=DEBUG)

    control_server = websockets.serve(control_handler, "0.0.0.0", CONTROL_PORT, max_size=None)
    stream_server = websockets.serve(stream_handler, "0.0.0.0", STREAM_PORT, max_size=None)

    async with control_server, stream_server:
        print(f"Control server: ws://0.0.0.0:{CONTROL_PORT}")
        print(f"Stream server:  ws://0.0.0.0:{STREAM_PORT}")
        print('Commands: {"cmd":"mode","mode":"servo"|"inspect"|"idle"}, {"cmd":"status"}, {"cmd":"stop"}')
        await asyncio.Future()


def main():
    devices = sl.Camera.get_device_list()
    if not devices:
        print("No ZED camera found")
        sys.exit(1)

    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if MANAGER is not None:
                MANAGER.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
