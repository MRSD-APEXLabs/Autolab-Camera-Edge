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
            y=y - 66.5,
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
            y=y - 66.5,
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
            y=y - 66.5,
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
        print("Doo")
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

        except Exception as e:
            print(e)
            self.exc = e
            raise
        finally:
            self.cleanup()


