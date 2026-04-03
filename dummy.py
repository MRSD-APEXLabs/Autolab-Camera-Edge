#!/usr/bin/env python3
"""
Unified controller for two mutually exclusive ZED camera modes.

Control websocket:
  ws://0.0.0.0:8765

Stream websocket:
  ws://0.0.0.0:8766

Control messages:
  {"cmd": "mode", "mode": "servo"}
  {"cmd": "mode", "mode": "inspect"}
  {"cmd": "mode", "mode": "idle"}
  {"cmd": "status"}
  {"cmd": "stop"}

Stream messages:
  When servo mode is active:
    {
      "type": "servo_frame",
      "frame_id": 12,
      "timestamp": 1234567890.123,
      "image_jpeg_b64": "...",
      "detections": [...],
      "target_uv": [u, v],
      "fps": 29.7
    }

  When inspect mode is active:
    {
      "type": "inspect_frame",
      "frame_id": 12,
      "timestamp": 1234567890.123,
      "image_jpeg_b64": "...",
      "pointcloud_zlib_b64": "...",
      "pointcloud_shape": [N, 4],
      "pointcloud_dtype": "float32",
      "wellplates": [...],
      "apriltags": [...],
      "pc_count": 12345,
      "fps": 14.2
    }

This file keeps your visual servo logic and adds streaming for both
modes so a client can visualize the live camera image in either case.
"""

import asyncio
import base64
import json
import sys
import threading
import time
import zlib
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pyzed.sl as sl
import websockets
from ultralytics import YOLO
from xarm.wrapper import XArmAPI
import apriltag


# =============================================================================
# Configuration
# =============================================================================

DEBUG = False

SERIAL_CAM1 = "40128964"   # visual servo camera
SERIAL_CAM2 = "42757821"   # inspect camera (video + pointcloud + apriltags + 2nd YOLO)
ARM_IP = "192.168.1.236"

MODEL_SERVO = "models/best_wrist.pt"
MODEL_INSPECT = "models/best_top.pt"   # set your second model path here

CONTROL_PORT = 8765
STREAM_PORT = 8766

AREA_STOP_THRESHOLD = 50000.0
START_POS_MM = np.array([-2.6, 320.0, 222.0], dtype=np.float64)
START_RPY_DEG = np.array([-180.0, 0.0, 0.0], dtype=np.float64)

POINTCLOUD_STRIDE = 8
STREAM_HZ = 15.0


# =============================================================================
# Utility functions
# =============================================================================

def wrap_angle_pi(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def draw_crosshair(img, uv, size=12, color=(0, 255, 255), thickness=2):
    u, v = int(uv[0]), int(uv[1])
    cv2.line(img, (u - size, v), (u + size, v), color, thickness)
    cv2.line(img, (u, v - size), (u, v + size), color, thickness)
    cv2.circle(img, (u, v), 4, color, -1)
    return img


def draw_velocity_arrow(img, center_uv, Vc, scale=600.0, color=(0, 0, 255), thickness=3):
    u, v = center_uv
    du = int(scale * Vc[1])
    dv = int(-scale * Vc[0])
    end = (int(u + du), int(v + dv))
    start = (int(u), int(v))
    cv2.arrowedLine(img, start, end, color, thickness, tipLength=0.25)

    mag = float(np.linalg.norm(Vc[:3]))
    cv2.putText(
        img,
        f"|v|={mag:.3f} m/s",
        (start[0] + 10, start[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )
    return img


def se3_to_adj(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    p = T[:3, 3]
    px = np.array(
        [
            [0.0, -p[2], p[1]],
            [p[2], 0.0, -p[0]],
            [-p[1], p[0], 0.0],
        ],
        dtype=np.float64,
    )

    Ad = np.zeros((6, 6), dtype=np.float64)
    Ad[:3, :3] = R
    Ad[:3, 3:] = px @ R
    Ad[3:, 3:] = R
    return Ad


def best_obb_from_results(results):
    r0 = results[0]
    if r0.obb is None or len(r0.obb) == 0:
        return None

    confs = r0.obb.conf.detach().cpu().numpy()
    idx = int(np.argmax(confs))

    xywhr = r0.obb.xywhr.detach().cpu().numpy()[idx]
    cls_id = int(r0.obb.cls.detach().cpu().numpy()[idx]) if r0.obb.cls is not None else -1
    conf = float(confs[idx])

    u = float(xywhr[0])
    v = float(xywhr[1])
    w = float(xywhr[2])
    h = float(xywhr[3])
    theta = float(xywhr[4])

    return u, v, w, h, theta, conf, cls_id


def all_obb_detections(results):
    r0 = results[0]
    if r0.obb is None or len(r0.obb) == 0:
        return []

    xywhr = r0.obb.xywhr.detach().cpu().numpy()
    confs = r0.obb.conf.detach().cpu().numpy()
    cls_ids = r0.obb.cls.detach().cpu().numpy() if r0.obb.cls is not None else np.full(len(confs), -1)
    names = getattr(r0, "names", {})

    dets = []
    for i in range(len(confs)):
        x, y, w, h, theta = map(float, xywhr[i])
        cls_id = int(cls_ids[i])
        dets.append(
            {
                "center": [x, y],
                "size": [w, h],
                "theta": theta,
                "conf": float(confs[i]),
                "cls_id": cls_id,
                "name": str(names.get(cls_id, cls_id)),
            }
        )
    return dets


def image_ibvs_command(obb, target_uv, desired_area=None, desired_theta=0.0):
    u, v, w, h, theta, conf, cls_id = obb
    ut, vt = target_uv

    ex = (v - vt) / max(vt + 1.0, 1.0)
    ey = (u - ut) / max(ut + 1.0, 1.0)

    area = max(w * h, 1.0)
    if desired_area is None:
        desired_area = area

    ez = (desired_area - area) / desired_area
    etheta = wrap_angle_pi(theta - desired_theta)

    kx = 0.6
    ky = 0.6
    kz = 0.8
    kw = 1.0

    Vc = np.array([kx * ex, -ky * ey, 0.25 * ez, 0.0, 0.0, -kw * etheta], dtype=np.float64)
    return Vc, (u, v, w, h, theta, conf, cls_id)


def encode_jpeg_b64(frame: np.ndarray, quality: int = 80) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("Failed to JPEG-encode frame")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def encode_zlib_b64(arr: np.ndarray) -> Tuple[str, List[int], str]:
    arr = np.asarray(arr)
    payload = zlib.compress(arr.tobytes(), level=6)
    return base64.b64encode(payload).decode("ascii"), list(arr.shape), str(arr.dtype)


# =============================================================================
# xArm helper functions
# =============================================================================

def clear_errors(arm):
    arm.clean_error()
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)
    arm.set_mode(1)
    arm.set_state(0)
    print("Errors cleared.")


def setup_gripper(arm):
    arm.set_tgpio_modbus_timeout(50)
    arm.set_tgpio_modbus_baudrate(115200)
    print("Gripper Modbus Configured.")


def gripper_open(arm):
    data = [0x08, 0x10, 0x07, 0x00, 0x00, 0x02, 0x04, 0x00, 0x00, 0x00, 0x00]
    code, ret = arm.getset_tgpio_modbus_data(data, is_transparent_transmission=False)
    print(f"OPEN Gripper: code={code}, ret={ret}")
    if code != 0:
        print("Error opening gripper, clearing errors...")
        clear_errors(arm)


def gripper_close(arm):
    data = [0x08, 0x10, 0x07, 0x00, 0x00, 0x02, 0x04, 0x00, 0x00, 0x00, 0x9E]
    code, ret = arm.getset_tgpio_modbus_data(data, is_transparent_transmission=False)
    print(f"CLOSE Gripper (158): code={code}, ret={ret}")
    if code != 0:
        print("Error closing gripper, clearing errors...")
        clear_errors(arm)


# =============================================================================
# Stream hub
# =============================================================================

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


# =============================================================================
# Worker A: visual servo + camera streaming
# =============================================================================

class ZEDYOLOServo(CameraWorker):
    def __init__(self, ip: str, serial: str, stream_hub: StreamHub, debug: bool = False):
        super().__init__("servo", serial)
        self.ip = ip
        self.debug = debug
        self.stream_hub = stream_hub
        self.camera = sl.Camera()
        self.runtime = sl.RuntimeParameters()
        self.view = sl.VIEW.RIGHT
        self.model = YOLO(MODEL_SERVO)
        self.model.to("cuda")
        self.threshold_action_done = False

        self.T_g_cr = np.array(
            [
                [1.000, 0.000, 0.000, 0.0],
                [0.000, 1.000, 0.000, 0.0],
                [0.000, 0.000, 1.000, 0.0],
                [0.000, 0.000, 0.000, 1.0],
            ],
            dtype=np.float64,
        )
        self.Ad_g_c = se3_to_adj(self.T_g_cr)

        init = sl.InitParameters()
        init.set_from_serial_number(serial)
        init.camera_resolution = sl.RESOLUTION.SVGA
        init.camera_fps = 30
        init.depth_mode = sl.DEPTH_MODE.NONE

        err = self.camera.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Camera {serial} failed to open: {err}")

        cam_info = self.camera.get_camera_information()
        right_calib = cam_info.camera_configuration.calibration_parameters.right_cam
        self.fx = float(right_calib.fx)
        self.fy = float(right_calib.fy)
        self.cx = float(right_calib.cx)
        self.cy = float(right_calib.cy)

        dummy = np.zeros((600, 800, 3), dtype=np.uint8)
        for _ in range(3):
            _ = self.model(dummy, verbose=False)
        
        if not self.debug:
            self.arm = XArmAPI(ip)
            self.arm.connect()
            if not self.arm.connected:
                raise RuntimeError(f"Failed to connect to xArm at {ip}")

            self.arm.motion_enable(True)
            self.arm.set_mode(0)
            self.arm.set_state(0)

            setup_gripper(self.arm)
            gripper_open(self.arm)

    def cleanup(self):
        try:
            self.stop_robot()
        except Exception:
            pass

        try:
            if not self.debug:
                self.arm.set_mode(0)
                self.arm.set_state(0)
        except Exception:
            pass

        try:
            self.camera.close()
        except Exception:
            pass

        try:
            cv2.destroyWindow("visual_servoing_right")
        except Exception:
            pass

    def execute_threshold_motion(self):
        if self.debug:
            return
        self.stop_robot()
        self.arm.motion_enable(True)
        self.arm.set_mode(0)
        self.arm.set_state(0)
        time.sleep(2)

        if self.debug:
            print("[DEBUG] Threshold reached: would execute post-threshold motion")
            return

        print("Threshold reached: executing post-threshold motion...")

        ret = self.arm.get_position(is_radian=False)
        if not isinstance(ret, tuple) or len(ret) < 2:
            raise RuntimeError(f"Unexpected get_position() result: {ret}")

        code, pose = ret[0], ret[1]
        if code != 0:
            raise RuntimeError(f"xArm get_position failed with code {code}")

        x, y, z, roll, pitch, yaw = pose[:6]
        print(
            f"Current pose: x={x:.1f} mm, y={y:.1f} mm, z={z:.1f} mm, "
            f"roll={roll:.1f} deg, pitch={pitch:.1f} deg, yaw={yaw:.1f} deg"
        )

        code = self.arm.set_position(
            x=x + 94.3,
            y=y - 54,
            z=z,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            speed=100,
            wait=True,
        )
        if code != 0:
            raise RuntimeError(f"xArm first threshold move failed with code {code}")

        code = self.arm.set_position(
            x=x + 94.3,
            y=y - 54,
            z=-52.7,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            speed=10,
            wait=True,
        )
        if code != 0:
            raise RuntimeError(f"xArm second threshold move failed with code {code}")

        gripper_close(self.arm)
        time.sleep(5)

        code = self.arm.set_position(
            x=x + 94.3,
            y=y - 54,
            z=0,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            speed=10,
            wait=True,
        )
        if code != 0:
            raise RuntimeError(f"xArm fourth threshold move failed with code {code}")

    def project_gripper_center(self, image_shape):
        H, W = image_shape[:2]
        p_g = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        p_c = self.T_g_cr @ p_g
        x, y, z = float(p_c[0]), float(p_c[1]), float(p_c[2])

        if z <= 1e-6:
            return (W * 0.5, H * 0.5)

        u = self.fx * (x / z) + self.cx
        v = self.fy * (y / z) + self.cy
        u = float(np.clip(u, 0, W - 1))
        v = float(np.clip(v, 0, H - 1))
        return (u, v)

    def move_to_start_pose(self):
        if self.debug:
            print(
                f"[DEBUG] Requested start pose: xyz_mm=({START_POS_MM[0]:.1f}, {START_POS_MM[1]:.1f}, {START_POS_MM[2]:.1f}), "
                f"rpy_deg=({START_RPY_DEG[0]:.1f}, {START_RPY_DEG[1]:.1f}, {START_RPY_DEG[2]:.1f})"
            )
            return

        arm = self.arm
        arm.motion_enable(True)
        arm.set_mode(0)
        arm.set_state(0)

        code = arm.set_position(
            x=START_POS_MM[0],
            y=START_POS_MM[1],
            z=START_POS_MM[2],
            roll=START_RPY_DEG[0],
            pitch=START_RPY_DEG[1],
            yaw=START_RPY_DEG[2],
            speed=100,
            wait=True,
        )
        if code != 0:
            raise RuntimeError(f"xArm set_position failed with code {code}")

    def enable_cartesian_velocity_mode(self):
        if self.debug:
            print("[DEBUG] switching to cartesian velocity mode")
            return

        self.arm.motion_enable(True)
        code = self.arm.set_mode(5)
        if code != 0:
            raise RuntimeError(f"xArm set_mode(5) failed with code {code}")
        code = self.arm.set_state(0)
        if code != 0:
            raise RuntimeError(f"xArm set_state(0) failed with code {code}")
        time.sleep(0.1)

    def send_velocity_to_robot(self, Vg: np.ndarray):
        Vg = np.asarray(Vg, dtype=np.float64).copy()
        Vg[:3] = np.clip(Vg[:3], -0.08, 0.08)
        Vg[3:] = np.clip(Vg[3:], -0.40, 0.40)

        cmd = np.array(
            [Vg[0] * 1000.0, Vg[1] * 1000.0, Vg[2] * 1000.0, Vg[3], Vg[4], Vg[5]],
            dtype=np.float64,
        )

        if self.debug:
            print(
                f"[DEBUG] cmd cart vel: vx={cmd[0]:+.1f} mm/s, vy={cmd[1]:+.1f} mm/s, "
                f"vz={cmd[2]:+.1f} mm/s, wx={cmd[3]:+.3f}, wy={cmd[4]:+.3f}, wz={cmd[5]:+.3f}"
            )
            return

        code = self.arm.vc_set_cartesian_velocity(
            cmd.tolist(),
            is_radian=True,
            is_tool_coord=False,
            duration=0,
        )
        if code != 0:
            raise RuntimeError(f"xArm vc_set_cartesian_velocity failed with code {code}")

    def stop_robot(self):
        if self.debug:
            print("[DEBUG] stop robot")
            return

        try:
            self.arm.vc_set_cartesian_velocity(
                [0, 0, 0, 0, 0, 0],
                is_radian=True,
                is_tool_coord=True,
                duration=0,
            )
        except Exception as e:
            print(f"[WARN] stop_robot failed: {e}")

    def run(self):
        try:
            self.move_to_start_pose()
            self.enable_cartesian_velocity_mode()

            prev_time = time.time()
            desired_area = None

            while not self.should_stop():
                err = self.camera.grab(self.runtime)
                if err != sl.ERROR_CODE.SUCCESS:
                    time.sleep(0.001)
                    continue

                t0 = time.time()
                right_frame = sl.Mat()
                self.camera.retrieve_image(right_frame, self.view)
                frame = cv2.cvtColor(right_frame.get_data(), cv2.COLOR_BGRA2BGR)
                t1 = time.time()

                results = self.model(frame, verbose=False)
                t2 = time.time()

                obb = best_obb_from_results(results)
                target_uv = self.project_gripper_center(frame.shape)
                detections = []

                if obb is not None:
                    detections.append(
                        {
                            "center": [float(obb[0]), float(obb[1])],
                            "size": [float(obb[2]), float(obb[3])],
                            "theta": float(obb[4]),
                            "conf": float(obb[5]),
                            "cls_id": int(obb[6]),
                        }
                    )

                    area = obb[2] * obb[3]

                    if area > AREA_STOP_THRESHOLD and not self.threshold_action_done:
                        self.threshold_action_done = True
                        cv2.putText(frame, f"threshold reached: area={area:.0f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                        # cv2.imshow("visual_servoing_right", frame)
                        cv2.waitKey(1)
                        self.execute_threshold_motion()
                        break

                    if desired_area is None:
                        desired_area = area

                    Vc, (u, v, w, h, theta, conf, cls_id) = image_ibvs_command(
                        obb=obb,
                        target_uv=target_uv,
                        desired_area=desired_area,
                        desired_theta=0.0,
                    )

                    Vg = self.Ad_g_c @ Vc
                    Vg[0] *= -1

                    Vg[:3] = np.clip(Vg[:3], -0.08, 0.08)
                    Vg[3:] = np.clip(Vg[3:], -0.40, 0.40)
                    Vg[0] = np.clip(Vg[0], -0.01, 0.01)
                    Vg[1] = np.clip(Vg[1], -0.01, 0.01)
                    Vg[2] = -abs(np.clip(Vg[2], -0.02, 0.02))

                    print(f"clamped Vg: vx={Vg[0]:+.3f}, vy={Vg[1]:+.3f}, vz={Vg[2]:+.3f}")
                    self.send_velocity_to_robot(Vg)

                    draw_crosshair(frame, target_uv, size=16, color=(0, 255, 255), thickness=2)
                    cv2.putText(frame, "gripper target", (int(target_uv[0]) + 18, int(target_uv[1]) - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.circle(frame, (int(u), int(v)), 6, (0, 255, 0), -1)
                    frame = draw_velocity_arrow(frame, (u, v), Vc, scale=800.0)
                    cv2.putText(frame, f"cls={cls_id} conf={conf:.2f} theta={theta:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                else:
                    self.stop_robot()
                    draw_crosshair(frame, target_uv, size=16, color=(0, 255, 255), thickness=2)
                    cv2.putText(frame, "no detection", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

                t3 = time.time()
                fps = 1.0 / max((t3 - prev_time), 1e-6)
                prev_time = t3

                img_b64 = encode_jpeg_b64(frame)
                self.stream_hub.publish(
                    "servo",
                    {
                        "type": "servo_frame",
                        "image_jpeg_b64": img_b64,
                        "detections": detections,
                        "target_uv": [float(target_uv[0]), float(target_uv[1])],
                        "fps": float(fps),
                        "capture_ms": float((t1 - t0) * 1000.0),
                        "inference_ms": float((t2 - t1) * 1000.0),
                        "post_ms": float((t3 - t2) * 1000.0),
                    },
                )

                print(
                    f"Capture: {(t1 - t0) * 1000.0:.2f} ms | Inference: {(t2 - t1) * 1000.0:.2f} ms | "
                    f"Post: {(t3 - t2) * 1000.0:.2f} ms | FPS: {fps:.2f}"
                )

                # cv2.imshow("visual_servoing_right", frame)
                # key = cv2.waitKey(1) & 0xFF
                # if key == 27:
                #     break

        except BaseException as e:
            self.exc = e
            raise
        finally:
            self.cleanup()


# =============================================================================
# Worker B: inspect mode (video + pointcloud + apriltags + second YOLO)
# =============================================================================

class ZEDInspectWorker(CameraWorker):
    def __init__(self, serial: str, stream_hub: StreamHub, debug: bool = False):
        super().__init__("inspect", serial)
        self.debug = debug
        self.stream_hub = stream_hub
        self.camera = sl.Camera()
        self.runtime = sl.RuntimeParameters()
        self.frame_mat = sl.Mat()
        self.depth_mat = sl.Mat()
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

        cam_info = self.camera.get_camera_information()
        calib = cam_info.camera_configuration.calibration_parameters.left_cam
        self.fx = float(calib.fx)
        self.fy = float(calib.fy)
        self.cx = float(calib.cx)
        self.cy = float(calib.cy)

        # Adjust these to match your real hardware
        self.apriltag_size_m = 0.0725   # tag side length in meters
        self.wellplate_size_m = np.array([0.12776, 0.08548, 0.015], dtype=np.float64)

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

    def _depth_to_3d(self, u: float, v: float, depth_m: float):
        if not np.isfinite(depth_m) or depth_m <= 0.0:
            return None
        x = (u - self.cx) / self.fx * depth_m
        y = (v - self.cy) / self.fy * depth_m
        z = depth_m
        return np.array([x, y, z], dtype=np.float64)

    def _depth_at(self, depth_img, u: float, v: float, half_window: int = 2):
        h, w = depth_img.shape[:2]
        x0 = max(0, int(round(u)) - half_window)
        x1 = min(w, int(round(u)) + half_window + 1)
        y0 = max(0, int(round(v)) - half_window)
        y1 = min(h, int(round(v)) + half_window + 1)

        patch = depth_img[y0:y1, x0:x1].astype(np.float32).reshape(-1)
        patch = patch[np.isfinite(patch)]
        patch = patch[patch > 0.0]

        if patch.size == 0:
            return None
        return float(np.median(patch))

    def _sample_points_in_box(self, depth_img, cx: float, cy: float, w: float, h: float, step: int = 4):
        h_img, w_img = depth_img.shape[:2]
        x0 = max(0, int(cx - w / 2))
        x1 = min(w_img - 1, int(cx + w / 2))
        y0 = max(0, int(cy - h / 2))
        y1 = min(h_img - 1, int(cy + h / 2))

        pts = []
        for yy in range(y0, y1 + 1, step):
            for xx in range(x0, x1 + 1, step):
                d = self._depth_at(depth_img, xx, yy, half_window=1)
                if d is None:
                    continue
                p = self._depth_to_3d(xx, yy, d)
                if p is not None and np.all(np.isfinite(p)):
                    pts.append(p)

        if len(pts) == 0:
            return None
        return np.asarray(pts, dtype=np.float64)

    def _normalize(self, v: np.ndarray):
        n = float(np.linalg.norm(v))
        if n < 1e-9:
            return None
        return v / n

    def _rotmat_to_quat(self, R: np.ndarray):
        # returns x, y, z, w
        tr = float(np.trace(R))
        if tr > 0.0:
            S = np.sqrt(tr + 1.0) * 2.0
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
                qw = (R[2, 1] - R[1, 2]) / S
                qx = 0.25 * S
                qy = (R[0, 1] + R[1, 0]) / S
                qz = (R[0, 2] + R[2, 0]) / S
            elif R[1, 1] > R[2, 2]:
                S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
                qw = (R[0, 2] - R[2, 0]) / S
                qx = (R[0, 1] + R[1, 0]) / S
                qy = 0.25 * S
                qz = (R[1, 2] + R[2, 1]) / S
            else:
                S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
                qw = (R[1, 0] - R[0, 1]) / S
                qx = (R[0, 2] + R[2, 0]) / S
                qy = (R[1, 2] + R[2, 1]) / S
                qz = 0.25 * S

        q = np.array([qx, qy, qz, qw], dtype=np.float64)
        q /= max(np.linalg.norm(q), 1e-12)
        return q.tolist()

    def _pose_dict(self, tvec: np.ndarray, R: np.ndarray):
        qx, qy, qz, qw = self._rotmat_to_quat(R)
        return {
            "position": [float(tvec[0]), float(tvec[1]), float(tvec[2])],
            "orientation": [float(qx), float(qy), float(qz), float(qw)],
        }

    def _estimate_wellplate_pose(self, det, depth_img):
        cx, cy = det["center"]
        w, h = det["size"]
        theta = float(det["theta"])

        pts = self._sample_points_in_box(depth_img, cx, cy, w, h, step=4)

        # Fallback to center depth if the patch is sparse
        center_d = self._depth_at(depth_img, cx, cy, half_window=3)
        center_3d = self._depth_to_3d(cx, cy, center_d) if center_d is not None else None

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
                    [syaw,  cyaw, 0.0],
                    [0.0,   0.0,   1.0],
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
        p1_d = self._depth_at(depth_img, cx - du * dir_img[0], cy - du * dir_img[1], half_window=2)
        p2_d = self._depth_at(depth_img, cx + du * dir_img[0], cy + du * dir_img[1], half_window=2)

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

    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        Reorder corners into a stable clockwise order starting near top-left.
        """
        corners = np.asarray(corners, dtype=np.float64).reshape(4, 2)
        c = corners.mean(axis=0)
        angles = np.arctan2(corners[:, 1] - c[1], corners[:, 0] - c[0])
        corners = corners[np.argsort(angles)]

        # rotate so first point is top-left-ish
        idx0 = int(np.argmin(corners[:, 0] + corners[:, 1]))
        corners = np.roll(corners, -idx0, axis=0)
        return corners

    def _estimate_apriltag_pose(self, tag_id: int, corners_2d: np.ndarray):
        """
        Return dict with position/orientation if pose is found, else None.
        """
        s = float(self.apriltag_size_m)

        obj_pts = np.array(
            [
                [-s / 2.0, -s / 2.0, 0.0],
                [ s / 2.0, -s / 2.0, 0.0],
                [ s / 2.0,  s / 2.0, 0.0],
                [-s / 2.0,  s / 2.0, 0.0],
            ],
            dtype=np.float32,
        )

        img_pts = self._order_corners(corners_2d).astype(np.float32)

        K = np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        ok, rvec, tvec = cv2.solvePnP(
            obj_pts,
            img_pts,
            K,
            None,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None

        R, _ = cv2.Rodrigues(rvec)

        q = self._rotmat_to_quat(R)
        return {
            "position": [float(tvec[0]), float(tvec[1]), float(tvec[2])],
            "orientation": [float(q[0]), float(q[1]), float(q[2]), float(q[3])],
        }

    def detect_apriltags(self, frame, depth_img):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = self.apriltag_detector.detect(gray)

        tag_payload = []
        for r in results:
            if r.tag_id not in (2,13):
                continue
            corners = np.array(r.corners, dtype=np.float64)

            pts = corners.astype(np.int32).reshape((-1, 1, 2))
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

            pose = self._estimate_apriltag_pose(int(r.tag_id), corners)

            # fallback if solvePnP fails
            if pose is None:
                d = self._depth_at(depth_img, r.center[0], r.center[1], half_window=3)
                p = self._depth_to_3d(r.center[0], r.center[1], d) if d is not None else None
                if p is not None:
                    pose = {
                        "position": [float(p[0]), float(p[1]), float(p[2])],
                        "orientation": [0.0, 0.0, 0.0, 1.0],
                    }

            tag_payload.append(
                {
                    "id": int(r.tag_id),
                    "center": [float(r.center[0]), float(r.center[1])],
                    "corners": [[float(x), float(y)] for x, y in corners],
                    "pose": pose,
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

                # depth aligned to LEFT image
                self.camera.retrieve_measure(self.depth_mat, sl.MEASURE.DEPTH)
                depth_img = self.depth_mat.get_data().astype(np.float32)

                results = self.model(frame, verbose=False)
                dets_2d = all_obb_detections(results)

                dets_3d = []
                for det in dets_2d:
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

                frame, tags_3d = self.detect_apriltags(frame, depth_img)

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
                        "wellplates": dets_3d,
                        "apriltags": tags_3d,
                        "pc_count": int(len(pc)),
                        "fps": float(fps),
                        "frame_frame": "zed_left_camera",
                    },
                )

        except BaseException as e:
            self.exc = e
            raise
        finally:
            self.cleanup()
            
            
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
