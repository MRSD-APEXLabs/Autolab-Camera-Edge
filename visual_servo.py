import logging
import threading
import time

import cv2
import numpy as np
import pyzed.sl as sl
from ultralytics import YOLO

from cam_worker import CameraWorker
from utils import *
from config import *
from streamhub import StreamHub

logger = logging.getLogger(__name__)

# =============================================================================
# Worker A: visual servo + camera streaming
# =============================================================================


class ZEDYOLOServo(CameraWorker):
    def __init__(
        self, ip: str, camera, grab_lock: threading.Lock, stream_hub: StreamHub, debug: bool = False
    ):
        super().__init__("servo", camera, grab_lock)
        self.ip = ip
        self.debug = debug
        self.stream_hub = stream_hub
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

        if not debug:
            from xarm.wrapper import XArmAPI
            self.arm = XArmAPI(ip)
            self.arm.connect()
            if not self.arm.connected:
                raise RuntimeError(f"Failed to connect to xArm at {ip}")
            self.arm.motion_enable(True)
            self.arm.set_mode(0)
            self.arm.set_state(0)
            setup_gripper(self.arm)
            gripper_open(self.arm)
        else:
            self.arm = None
            logger.info("[DEBUG] arm skipped — debug mode")

    def cleanup(self):
        try:
            self.stop_robot()
        except Exception:
            pass

        try:
            if self.arm is not None:
                self.arm.set_mode(0)
                self.arm.set_state(0)
        except Exception:
            pass

        try:
            cv2.destroyWindow("visual_servoing_right")
        except Exception:
            pass

    def execute_threshold_motion(self):
        if self.debug:
            logger.info("[DEBUG] Threshold reached: would execute post-threshold motion")
            return

        self.stop_robot()
        self.arm.motion_enable(True)
        self.arm.set_mode(0)
        self.arm.set_state(0)
        time.sleep(2)

        logger.info("Threshold reached: executing post-threshold motion...")

        ret = self.arm.get_position(is_radian=False)
        if not isinstance(ret, tuple) or len(ret) < 2:
            raise RuntimeError(f"Unexpected get_position() result: {ret}")

        code, pose = ret[0], ret[1]
        if code != 0:
            raise RuntimeError(f"xArm get_position failed with code {code}")

        x, y, z, roll, pitch, yaw = pose[:6]
        logger.info(
            "Current pose: x=%.1f mm, y=%.1f mm, z=%.1f mm, roll=%.1f deg, pitch=%.1f deg, yaw=%.1f deg",
            x, y, z, roll, pitch, yaw,
        )

        code = self.arm.set_position(
            x=x + 94.3,
            y=y - 52,
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
            y=y - 52,
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
            y=y - 52,
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
            logger.info(
                "[DEBUG] Requested start pose: xyz_mm=(%.1f, %.1f, %.1f), rpy_deg=(%.1f, %.1f, %.1f)",
                START_POS_MM[0], START_POS_MM[1], START_POS_MM[2],
                START_RPY_DEG[0], START_RPY_DEG[1], START_RPY_DEG[2],
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
            logger.info("[DEBUG] switching to cartesian velocity mode")
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
            logger.debug(
                "[DEBUG] cmd cart vel: vx=%+.1f mm/s, vy=%+.1f mm/s, vz=%+.1f mm/s, wx=%+.3f, wy=%+.3f, wz=%+.3f",
                cmd[0], cmd[1], cmd[2], cmd[3], cmd[4], cmd[5],
            )
            return

        code = self.arm.vc_set_cartesian_velocity(
            cmd.tolist(),
            is_radian=True,
            is_tool_coord=False,
            duration=0,
        )
        if code != 0:
            raise RuntimeError(
                f"xArm vc_set_cartesian_velocity failed with code {code}"
            )

    def stop_robot(self):
        if self.debug:
            logger.debug("[DEBUG] stop robot")
            return

        try:
            self.arm.vc_set_cartesian_velocity(
                [0, 0, 0, 0, 0, 0],
                is_radian=True,
                is_tool_coord=True,
                duration=0,
            )
        except Exception as e:
            logger.warning("stop_robot failed: %s", e)

    def run(self):
        logger.info("servo worker starting")
        try:
            # Clear arm errors
            self.arm.clean_error()
            self.arm.clean_warn()
            time.sleep(0.1)

            self.move_to_start_pose()
            self.enable_cartesian_velocity_mode()

            prev_time = time.time()
            desired_area = None

            while not self.should_stop():
                right_frame = sl.Mat()
                with self.grab_lock:
                    err = self.camera.grab(self.runtime)
                    if err == sl.ERROR_CODE.SUCCESS:
                        self.camera.retrieve_image(right_frame, self.view)

                if err != sl.ERROR_CODE.SUCCESS:
                    time.sleep(0.001)
                    continue

                t0 = time.time()
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
                        cv2.putText(
                            frame,
                            f"threshold reached: area={area:.0f}",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 0, 255),
                            2,
                        )
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

                    logger.debug("clamped Vg: vx=%+.3f, vy=%+.3f, vz=%+.3f", Vg[0], Vg[1], Vg[2])
                    self.send_velocity_to_robot(Vg)

                    draw_crosshair(
                        frame, target_uv, size=16, color=(0, 255, 255), thickness=2
                    )
                    cv2.putText(
                        frame,
                        "gripper target",
                        (int(target_uv[0]) + 18, int(target_uv[1]) - 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )
                    cv2.circle(frame, (int(u), int(v)), 6, (0, 255, 0), -1)
                    frame = draw_velocity_arrow(frame, (u, v), Vc, scale=800.0)
                    cv2.putText(
                        frame,
                        f"cls={cls_id} conf={conf:.2f} theta={theta:.2f}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2,
                    )
                else:
                    self.stop_robot()
                    draw_crosshair(
                        frame, target_uv, size=16, color=(0, 255, 255), thickness=2
                    )
                    cv2.putText(
                        frame,
                        "no detection",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        2,
                    )

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

                logger.debug(
                    "capture=%.1fms  inference=%.1fms  post=%.1fms  fps=%.1f",
                    (t1 - t0) * 1000.0, (t2 - t1) * 1000.0, (t3 - t2) * 1000.0, fps,
                )

        except Exception as e:
            logger.exception("servo worker crashed: %s", e)
            self.exc = e
            raise
        finally:
            self.cleanup()
