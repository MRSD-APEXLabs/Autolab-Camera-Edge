import logging
import threading
import time

import cv2
import pyzed.sl as sl

from streamhub import StreamHub
from utils import encode_jpeg_b64

logger = logging.getLogger(__name__)


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
        self.base_runtime.enable_depth = False  # raw stream needs image only, not NEURAL depth
        self.wrist_mat = sl.Mat()
        self.base_mat = sl.Mat()

    def stop(self):
        self.stop_event.set()

    def run(self):
        logger.info("raw image worker starting (wrist + base)")
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

            time.sleep(0)  # yield so inspect/servo workers can acquire base_lock

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

            time.sleep(0)  # yield so inspect/servo workers can acquire wrist_lock

            if not wrist_ok and not base_ok:
                time.sleep(0.001)
