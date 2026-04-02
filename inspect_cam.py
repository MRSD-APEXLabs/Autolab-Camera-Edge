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
from cam_worker import CameraWorker
from utils import *
from config import *
from streamhub import StreamHub

# =============================================================================
# Worker B: pointcloud + apriltags + second YOLO
# =============================================================================

class ZEDInspectWorker(CameraWorker):
    def __init__(self, serial: str, stream_hub: StreamHub, debug: bool = False):
        super().__init__("inspect", serial)
        self.debug = debug
        self.stream_hub = stream_hub
        self.camera = sl.Camera()
        self.runtime = sl.RuntimeParameters()
        self.frame_mat = sl.Mat()
        self.pc_mat = sl.Mat()
        self.model = YOLO(MODEL_INSPECT)
        self.model.to("cuda")
        self.apriltag_detector = apriltag.Detector(apriltag.DetectorOptions(families="tag16h5"))

        init = sl.InitParameters()
        init.set_from_serial_number(serial)
        init.camera_resolution = sl.RESOLUTION.SVGA
        init.camera_fps = 30
        init.depth_mode = sl.DEPTH_MODE.NEURAL
        init.coordinate_units = sl.UNIT.METER
        init.depth_minimum_distance = 0.2
        init.depth_maximum_distance = 1.7

        err = self.camera.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Camera {serial} failed to open: {err}")

        dummy = np.zeros((600, 800, 3), dtype=np.uint8)
        for _ in range(3):
            _ = self.model(dummy, verbose=False)

    def cleanup(self):
        try:
            self.camera.close()
        except Exception:
            pass

        try:
            cv2.destroyWindow("inspect_camera")
        except Exception:
            pass

    def detect_apriltags(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = self.apriltag_detector.detect(gray)

        tag_payload = []
        for r in results:
            pts = np.array(r.corners, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(frame, str(r.tag_id), (int(r.center[0]), int(r.center[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            tag_payload.append(
                {
                    "id": int(r.tag_id),
                    "center": [float(r.center[0]), float(r.center[1])],
                    "corners": [[float(x), float(y)] for x, y in r.corners],
                }
            )

        return frame, tag_payload

    def run(self):
        try:
            while not self.should_stop():
                start = time.time()
                err = self.camera.grab(self.runtime)
                if err != sl.ERROR_CODE.SUCCESS:
                    time.sleep(0.001)
                    continue

                self.camera.retrieve_image(self.frame_mat, sl.VIEW.LEFT)
                frame = cv2.cvtColor(self.frame_mat.get_data(), cv2.COLOR_BGRA2BGR)

                results = self.model(frame, verbose=False)
                dets = all_obb_detections(results)

                for det in dets:
                    cx, cy = det["center"]
                    w, h = det["size"]
                    theta = det["theta"]
                    conf = det["conf"]
                    label = det["name"]
                    cv2.circle(frame, (int(cx), int(cy)), 5, (255, 255, 0), -1)
                    cv2.putText(frame, f"{label} {conf:.2f}", (int(cx) + 8, int(cy) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(frame, f"w={w:.0f} h={h:.0f} th={theta:.2f}", (int(cx) + 8, int(cy) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                frame, tags = self.detect_apriltags(frame)

                self.camera.retrieve_measure(self.pc_mat, sl.MEASURE.XYZRGBA)
                pc = self.pc_mat.get_data().reshape(-1, 4).astype(np.float32)
                valid = np.isfinite(pc[:, 0]) & np.isfinite(pc[:, 1]) & np.isfinite(pc[:, 2])
                pc = pc[valid]
                if POINTCLOUD_STRIDE > 1 and len(pc) > 0:
                    pc = pc[::POINTCLOUD_STRIDE]

                pc_b64, pc_shape, pc_dtype = encode_zlib_b64(pc)
                img_b64 = encode_jpeg_b64(frame)

                fps = 1.0 / max((time.time() - start), 1e-6)

                self.stream_hub.publish(
                    "inspect",
                    {
                        "type": "inspect_frame",
                        "image_jpeg_b64": img_b64,
                        "pointcloud_zlib_b64": pc_b64,
                        "pointcloud_shape": pc_shape,
                        "pointcloud_dtype": pc_dtype,
                        "wellplates": dets,
                        "apriltags": tags,
                        "pc_count": int(len(pc)),
                        "fps": float(fps),
                    },
                )

                # cv2.imshow("inspect_camera", frame)
                # key = cv2.waitKey(1) & 0xFF
                # if key == 27:
                #     break

        except BaseException as e:
            self.exc = e
            raise
        finally:
            self.cleanup()
