
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

# =============================================================================
# Base worker
# =============================================================================

class CameraWorker(threading.Thread):
    def __init__(self, name: str, serial: str):
        super().__init__(daemon=True)
        self.name = name
        self.serial = serial
        self.stop_event = threading.Event()
        self.exc: Optional[BaseException] = None

    def stop(self):
        self.stop_event.set()

    def should_stop(self) -> bool:
        return self.stop_event.is_set()

    def run(self):
        raise NotImplementedError

