
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
from config import *



class StreamHub:
    """Thread-safe latest-frame store for both modes."""

    def __init__(self):
        self._lock = threading.Lock()
        self._seq = {"servo": 0, "inspect": 0}
        self._latest: Dict[str, Optional[Dict[str, Any]]] = {"servo": None, "inspect": None}
        self._active_mode = "idle"

    def set_active_mode(self, mode: str):
        with self._lock:
            self._active_mode = mode

    def get_active_mode(self) -> str:
        with self._lock:
            return self._active_mode

    def publish(self, channel: str, payload: Dict[str, Any]):
        if channel not in self._latest:
            raise ValueError(f"Unknown channel: {channel}")
        with self._lock:
            self._seq[channel] += 1
            payload = dict(payload)
            payload["frame_id"] = self._seq[channel]
            payload["timestamp"] = time.time()
            self._latest[channel] = payload

    def snapshot(self, channel: str) -> Tuple[int, Optional[Dict[str, Any]]]:
        if channel not in self._latest:
            raise ValueError(f"Unknown channel: {channel}")
        with self._lock:
            if self._latest[channel] is None:
                return self._seq[channel], None
            return self._seq[channel], dict(self._latest[channel])


# class StreamHub:
#     """Thread-safe latest-frame store shared between the inspect worker and websocket server."""

#     def __init__(self):
#         self._lock = threading.Lock()
#         self._seq = 0
#         self._latest: Optional[Dict[str, Any]] = None

#     def publish(self, payload: Dict[str, Any]):
#         with self._lock:
#             self._seq += 1
#             payload = dict(payload)
#             payload["frame_id"] = self._seq
#             payload["timestamp"] = time.time()
#             self._latest = payload

#     def snapshot(self) -> Tuple[int, Optional[Dict[str, Any]]]:
#         with self._lock:
#             if self._latest is None:
#                 return self._seq, None
#             return self._seq, dict(self._latest)

