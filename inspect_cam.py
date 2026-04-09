import logging
import threading
import time

import cv2
import numpy as np
import pyzed.sl as sl
import apriltag
from ultralytics import YOLO

from cam_worker import CameraWorker
from utils import *
from config import *
from streamhub import StreamHub

logger = logging.getLogger(__name__)

# =============================================================================
# Worker B: pointcloud + apriltags + second YOLO
# =============================================================================


class ZEDInspectWorker(CameraWorker):
    def __init__(self, camera, grab_lock: threading.Lock, stream_hub: StreamHub,
                 debug: bool = False):
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

        self._run_event = threading.Event()
        self._on_run_complete = None

    # -------------------------------------------------------------------------
    # State machine control
    # -------------------------------------------------------------------------

    def start_run(self, on_complete=None):
        self._on_run_complete = on_complete
        self._run_event.set()

    def abort_run(self):
        self._run_event.clear()

    def stop(self):
        """Kill the worker thread entirely."""
        self.stop_event.set()
        self._run_event.set()  # unblock wait so thread can exit

    def should_stop(self) -> bool:
        return not self._run_event.is_set() or self.stop_event.is_set()

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def cleanup(self):
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
            cv2.putText(
                frame,
                str(r.tag_id),
                (int(r.center[0]), int(r.center[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            tag_payload.append(
                {
                    "id": int(r.tag_id),
                    "center": [float(r.center[0]), float(r.center[1])],
                    "corners": [[float(x), float(y)] for x, y in r.corners],
                }
            )

        return frame, tag_payload

    def _estimate_wellplate_pose(self, det, depth_img):
        cx, cy = det["center"]
        w, h = det["size"]
        theta = float(det["theta"])

        pts = self._sample_points_in_box(depth_img, cx, cy, w, h, step=4)

        # Fallback to center depth if the patch is sparse
        center_d = self._depth_at(depth_img, cx, cy, half_window=3)
        center_3d = (
            self._depth_to_3d(cx, cy, center_d) if center_d is not None else None
        )

        if pts is None or len(pts) < 20:
            if center_3d is None:
                return None

            # Use image theta as yaw, depth gives translation
            yaw = -theta
            cyaw = np.cos(yaw)
            syaw = np.sin(yaw)
            R = np.array(
                [
                    [cyaw, -syaw, 0.0],
                    [syaw, cyaw, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            return self._pose_dict(center_3d, R)

        # Fit a local plane with PCA
        centroid = pts.mean(axis=0)
        Q = pts - centroid
        _, _, vh = np.linalg.svd(Q, full_matrices=False)

        # Plane normal = smallest singular vector
        normal = vh[-1]
        normal = self._normalize(normal)
        if normal is None:
            return None

        # Keep normal pointing toward camera
        if normal[2] > 0.0:
            normal = -normal

        # Derive an in-plane x-axis from the 2D OBB angle
        dir_img = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
        du = max(8.0, 0.25 * max(w, h))
        p1_d = self._depth_at(
            depth_img, cx - du * dir_img[0], cy - du * dir_img[1], half_window=2
        )
        p2_d = self._depth_at(
            depth_img, cx + du * dir_img[0], cy + du * dir_img[1], half_window=2
        )

        x_axis = None
        if p1_d is not None and p2_d is not None:
            p1 = self._depth_to_3d(cx - du * dir_img[0], cy - du * dir_img[1], p1_d)
            p2 = self._depth_to_3d(cx + du * dir_img[0], cy + du * dir_img[1], p2_d)
            if p1 is not None and p2 is not None:
                x_axis = p2 - p1
                x_axis = x_axis - np.dot(x_axis, normal) * normal
                x_axis = self._normalize(x_axis)

        if x_axis is None:
            # fallback: principal direction in the plane
            S = Q - np.outer(Q @ normal, normal)
            cov = np.cov(S.T)
            evals, evecs = np.linalg.eigh(cov)
            x_axis = evecs[:, np.argmax(evals)]
            x_axis = x_axis - np.dot(x_axis, normal) * normal
            x_axis = self._normalize(x_axis)

        if x_axis is None:
            return None

        y_axis = np.cross(normal, x_axis)
        y_axis = self._normalize(y_axis)
        if y_axis is None:
            return None

        x_axis = np.cross(y_axis, normal)
        x_axis = self._normalize(x_axis)
        if x_axis is None:
            return None

        R = np.column_stack([x_axis, y_axis, normal])

        # Use the plane centroid as the 3D location
        tvec = centroid.reshape(3)
        return self._pose_dict(tvec, R)

    # -------------------------------------------------------------------------
    # Thread entry points
    # -------------------------------------------------------------------------

    def _execute_run(self):
        logger.info("inspect run starting")

        while not self.should_stop():
            start = time.time()

            with self.grab_lock:
                err = self.camera.grab(self.runtime)
                if err == sl.ERROR_CODE.SUCCESS:
                    self.camera.retrieve_image(self.frame_mat, sl.VIEW.LEFT)
                    self.camera.retrieve_measure(self.pc_mat, sl.MEASURE.XYZRGBA)

            if err != sl.ERROR_CODE.SUCCESS:
                time.sleep(0.001)
                continue

            frame = cv2.cvtColor(self.frame_mat.get_data(), cv2.COLOR_BGRA2BGR)

            results = self.model(frame, verbose=False)
            dets = all_obb_detections(results)

            dets_3d = []
            for det in dets:
                cx, cy = det["center"]
                w, h = det["size"]
                theta = det["theta"]
                conf = det["conf"]
                label = det["name"]
                cv2.circle(frame, (int(cx), int(cy)), 5, (255, 255, 0), -1)
                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (int(cx) + 8, int(cy) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"w={w:.0f} h={h:.0f} th={theta:.2f}",
                    (int(cx) + 8, int(cy) + 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                )

                pose = self._estimate_wellplate_pose(det, depth_img)
                det_3d = dict(det)
                det_3d["pose"] = pose
                dets_3d.append(det_3d)

            frame, tags = self.detect_apriltags(frame)

            pc = self.pc_mat.get_data().reshape(-1, 4).astype(np.float32)
            valid = (
                np.isfinite(pc[:, 0])
                & np.isfinite(pc[:, 1])
                & np.isfinite(pc[:, 2])
            )
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
                    "wellplates": dets_3d,
                    "apriltags": tags,
                    "pc_count": int(len(pc)),
                    "fps": float(fps),
                },
            )
            logger.debug(
                "fps=%.1f  wellplates=%d  apriltags=%d  pts=%d",
                fps, len(dets), len(tags), len(pc),
            )

        logger.info("inspect run finished")

    def _run(self):
        logger.info("inspect worker ready")
        while not self.stop_event.is_set():
            if not self._run_event.wait(timeout=0.5):
                continue
            if self.stop_event.is_set():
                break
            try:
                self._execute_run()
            except Exception as e:
                logger.exception("inspect worker crashed: %s", e)
                self.exc = e
            finally:
                self.cleanup()
                self._run_event.clear()
                if self._on_run_complete:
                    self._on_run_complete()
