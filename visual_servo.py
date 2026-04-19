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
        self, arm, camera, grab_lock: threading.Lock, stream_hub: StreamHub,
        debug: bool = False
    ):
        super().__init__("servo", camera, grab_lock)
        self.arm = arm
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

        self._run_event = threading.Event()
        self._on_run_complete = None

    # -------------------------------------------------------------------------
    # State machine control
    # -------------------------------------------------------------------------

    def start_run(self, on_complete=None):
        self.threshold_action_done = False
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
    # Robot helpers
    # -------------------------------------------------------------------------

    def cleanup(self):
        try:
            self.stop_robot()
        except Exception:
            pass

        try:
            if self.arm is not None:
                # Use mode 1 (online trajectory planning) so MoveIt/ROS2 can
                # reactivate its controller after servo exits.  Mode 0 leaves
                # the arm in a state the xarm_ros2 driver cannot take over from.
                self.arm.clean_error()
                self.arm.clean_warn()
                self.arm.set_mode(1)
                self.arm.set_state(0)
                self.arm.disconnect()
        except Exception:
            pass

        try:
            cv2.destroyWindow("visual_servoing_right")
        except Exception:
            pass

    def execute_threshold_motion(self, obb=None):
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
        roll = 179
        pitch = 0
        logger.info(
            "Current pose: x=%.1f mm, y=%.1f mm, z=%.1f mm, roll=%.1f deg, pitch=%.1f deg, yaw=%.1f deg",
            x, y, z, roll, pitch, yaw,
        )

        if obb is not None:
            _, _, w_px, h_px, _, _, _ = obb
            pixel_area = w_px * h_px
            if pixel_area > 0:
                # Pinhole model: Z_mm = sqrt(fx * fy * physical_area_mm2 / pixel_area_px2)
                estimated_depth_mm = float(
                    np.sqrt(self.fx * self.fy * PLATE_WIDTH_MM * PLATE_HEIGHT_MM / pixel_area)
                )
                descent_mm = max(10.0, estimated_depth_mm - CAMERA_TO_GRASP_OFFSET_MM)
                logger.info(
                    "OBB depth estimate: pixel_area=%.0f px² → depth=%.1f mm → descent=%.1f mm",
                    pixel_area, estimated_depth_mm, descent_mm,
                )
            else:
                descent_mm = GRASP_DESCENT_MM
                logger.warning("OBB pixel area is zero, using fallback GRASP_DESCENT_MM=%.1f mm", descent_mm)
        else:
            descent_mm = GRASP_DESCENT_MM
            logger.warning("No OBB at threshold, using fallback GRASP_DESCENT_MM=%.1f mm", descent_mm)

        z_grasp = -0.8

        # --- Compute grasp yaw to align gripper with plate's shorter axis ---
        # (fingers contact the midpoints of the two longer edges)
        # IBVS converges with the plate's w-axis horizontal (theta ≈ 0).
        #   w >= h → longer axis is horizontal (along theta) → shorter axis at theta + 90°
        #   w <  h → longer axis is vertical (at theta + 90°) → shorter axis at theta
        # From the IBVS sign convention (Vc[5] = -kw*theta), the yaw delta is the
        # negative of the target shorter-axis image angle.  GRASP_YAW_OFFSET_DEG
        # trims any residual physical mounting offset.
        grasp_yaw = yaw
        if obb is not None:
            _, _, w_obb, h_obb, theta_obb, _, _ = obb
            if w_obb >= h_obb:
                shorter_axis_rad = theta_obb + np.pi / 2
            else:
                shorter_axis_rad = theta_obb
            yaw_delta_deg = -float(np.degrees(shorter_axis_rad)) + GRASP_YAW_OFFSET_DEG
            grasp_yaw = yaw - yaw_delta_deg
            logger.info(
                "Plate OBB: w=%.1f h=%.1f theta=%.3f rad → yaw_delta=%.1f° → grasp_yaw=%.1f°",
                w_obb, h_obb, theta_obb, yaw_delta_deg, grasp_yaw,
            )

        # The nominal offset (dx, dy) was calibrated at the starting yaw.
        # When grasp_yaw != yaw the end-effector rotates, so rotate the offset
        # vector by the same delta to keep it aligned in the world frame.
        # dx_nom, dy_nom = 94.3, -66.5

        dx, dy = (92, -62) if abs(grasp_yaw - yaw) < 10 else (88, -55)
        z_grasp = -50 if abs(grasp_yaw - yaw) < 10 else z_grasp
        if y < 0:
            z_grasp=-17
        # yaw_delta_rad = np.radians(grasp_yaw - yaw)
        # cos_d, sin_d = np.cos(yaw_delta_rad), np.sin(yaw_delta_rad)
        # dx = cos_d * dx_nom - sin_d * dy_nom
        # dy = sin_d * dx_nom + cos_d * dy_nom
        # logger.info(
        #     "Grasp offset: nominal=(%.1f, %.1f) → rotated=(%.1f, %.1f) mm (yaw_delta=%.1f°)",
        #     dx_nom, dy_nom, dx, dy, np.degrees(yaw_delta_rad),
        # )
        print(y, z_grasp)

        code = self.arm.set_position(
            x=x + dx,
            y=y + dy,
            z=z,
            roll=roll,
            pitch=pitch,
            yaw=grasp_yaw,
            speed=100,
            wait=True,
        )
        if code != 0:
            raise RuntimeError(f"xArm first threshold move failed with code {code}")

        code = self.arm.set_position(
            x=x + dx,
            y=y + dy,
            z=z_grasp,
            roll=roll,
            pitch=pitch,
            yaw=grasp_yaw,
            speed=10,
            wait=True,
        )
        if code != 0:
            raise RuntimeError(f"xArm second threshold move failed with code {code}")

        fsr1, fsr2, r1, r2 = None, None, None, None
        try:
            samples = [gripper_get_fsr(self.arm) for _ in range(FSR_BASELINE_SAMPLES)]
            valid = [(f1, f2) for f1, f2 in samples if f1 is not None]
            base1 = int(sum(f1 for f1, _ in valid) / len(valid)) if valid else 0
            base2 = int(sum(f2 for _, f2 in valid) / len(valid)) if valid else 0
            logger.info("FSR baseline (%d samples): fsr1=%d fsr2=%d", len(valid), base1, base2)

            yaw_rad = np.radians(grasp_yaw)
            c, s = np.cos(yaw_rad), np.sin(yaw_rad)
            world_dir = np.array([c, s, 0.0]) if FSR_FINGER_TOOL_AXIS == "x" else np.array([-s, c, 0.0])

            for close_attempt in range(FSR_CLOSE_NUDGE_RETRIES + 1):
                if close_attempt > 0:
                    logger.info("Grasp threshold not met — gripper close+nudge (attempt %d/%d)",
                                close_attempt, FSR_CLOSE_NUDGE_RETRIES)

                gripper_close(self.arm)

                # re-assert mode 0 so arm accepts position commands while gripper closes
                self.arm.clean_error()
                self.arm.motion_enable(True)
                self.arm.set_mode(0)
                self.arm.set_state(0)
                time.sleep(0.2)

                deadline = time.time() + 7.0
                while time.time() < deadline:
                    fsr1, fsr2 = gripper_get_fsr(self.arm)
                    if fsr1 is None:
                        time.sleep(0.1)
                        continue

                    r1 = fsr1 - base1
                    r2 = fsr2 - base2
                    logger.info("FSR during close: fsr1=%d fsr2=%d  (relative: r1=%d r2=%d)", fsr1, fsr2, r1, r2)
                    delta = r1 - r2
                    if abs(delta) > FSR_NUDGE_MIN_DELTA:
                        signed_mm = -FSR_NUDGE_MM if delta > 0 else +FSR_NUDGE_MM
                        ret = self.arm.get_position(is_radian=False)
                        if isinstance(ret, tuple) and len(ret) >= 2 and ret[0] == 0:
                            cx, cy, cz = ret[1][0], ret[1][1], ret[1][2]
                            d = world_dir * signed_mm
                            self.arm.set_position(
                                x=cx + d[0], y=cy + d[1], z=cz + d[2],
                                roll=roll, pitch=pitch, yaw=grasp_yaw,
                                speed=20, wait=True,
                            )
                    else:
                        time.sleep(0.1)

                r1 = (fsr1 - base1) if fsr1 is not None else 0
                r2 = (fsr2 - base2) if fsr2 is not None else 0
                logger.info("Post-close FSR (attempt %d): fsr1=%s fsr2=%s  relative: r1=%d r2=%d",
                            close_attempt, fsr1, fsr2, r1, r2)

                if r1 > FSR_GRASP_THRESHOLD and r2 > FSR_GRASP_THRESHOLD:
                    logger.info("Grasp threshold met on attempt %d", close_attempt)
                    break
                if close_attempt == FSR_CLOSE_NUDGE_RETRIES:
                    logger.warning("Grasp threshold not met after %d attempts", FSR_CLOSE_NUDGE_RETRIES + 1)
        except Exception as e:
            logger.warning("gripper_close failed (no gripper?): %s", e)


        # Re-assert mode 0 before moving back up: gripper_close (or gripping a
        # physical object) can leave the arm in an error/mode-1 state.
        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(True)
        self.arm.set_mode(0)
        self.arm.set_state(0)
        time.sleep(0.5)

        code = self.arm.set_position(
            x=x + dx,
            y=y + dy,
            z=z,
            roll=roll,
            pitch=pitch,
            yaw=grasp_yaw,
            speed=50,
            wait=True,
        )
        if code != 0:
            logger.warning("xArm fourth threshold move failed with code %d", code)

        code = self.arm.set_position(
            x=x + dx,
            y=y + dy,
            z=z,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            speed=50,
            wait=True,
        )
        if code != 0:
            logger.warning("xArm fifth threshold move failed with code %d", code)

        return r1, r2

    def _lift_for_retry(self):
        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(True)
        self.arm.set_mode(0)
        self.arm.set_state(0)
        time.sleep(0.3)

        ret = self.arm.get_position(is_radian=False)
        if not isinstance(ret, tuple) or len(ret) < 2 or ret[0] != 0:
            raise RuntimeError(f"get_position failed during retry lift: {ret}")
        x, y, z, roll, pitch, yaw = ret[1][:6]

        code = self.arm.set_position(
            x=x, y=y, z=z + GRASP_RETRY_LIFT_MM,
            roll=roll, pitch=pitch, yaw=yaw,
            speed=100, wait=True,
        )
        if code != 0:
            raise RuntimeError(f"retry lift failed with code {code}")

        gripper_open(self.arm)
        logger.info("Retry lift complete: z %.1f → %.1f mm", z, z + GRASP_RETRY_LIFT_MM)

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
        arm.clean_error()
        arm.clean_warn()
        arm.motion_enable(True)
        arm.set_mode(0)
        arm.set_state(0)
        time.sleep(0.1)
        gripper_open(self.arm)

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

        roll = 180
        pitch = 0

        code = self.arm.set_position(
            x=x,
            y=y,
            z=z,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            speed=100,
            wait=True,
        )

        if code != 0:
            raise RuntimeError(f"xArm alignment failed with code {code}")

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

    # -------------------------------------------------------------------------
    # Thread entry points
    # -------------------------------------------------------------------------

    def _execute_run(self):
        logger.info("servo run starting")

        if not self.debug:
            setup_gripper(self.arm)
            gripper_open(self.arm)

        self.arm.connect()
        self.move_to_start_pose()
        self.enable_cartesian_velocity_mode()

        prev_time = time.time()

        for attempt in range(GRASP_MAX_RETRIES + 1):
            desired_area = None
            self.threshold_action_done = False
            grasp_succeeded = False

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
                        if not self.debug:
                            fsr1, fsr2 = self.execute_threshold_motion(obb=obb)
                            if fsr1 is not None and (fsr1 > FSR_GRASP_THRESHOLD and fsr2 > FSR_GRASP_THRESHOLD):
                                logger.info(
                                    "Grasp check: fsr1=%d fsr2=%d > threshold=%d — succeeded",
                                    fsr1, fsr2, FSR_GRASP_THRESHOLD,
                                )
                                grasp_succeeded = True
                            else:
                                logger.warning(
                                    "Grasp check: fsr1=%s fsr2=%s <= threshold=%d — failed (empty gripper)",
                                    fsr1, fsr2, FSR_GRASP_THRESHOLD,
                                )
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
                    Vg[0] = np.clip(Vg[0], -0.02, 0.02)
                    Vg[1] = np.clip(Vg[1], -0.02, 0.02)
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
                    draw_obb(frame, u, v, w, h, theta, color=(0, 255, 0), thickness=2)
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

            # --- post-servo-loop: retry logic ---
            if self.should_stop():
                break

            if grasp_succeeded or self.debug:
                break

            if attempt < GRASP_MAX_RETRIES:
                logger.info("Grasp failed — retry %d/%d: lifting %.0f mm and re-running servo",
                            attempt + 1, GRASP_MAX_RETRIES, GRASP_RETRY_LIFT_MM)
                self._lift_for_retry()
                self.enable_cartesian_velocity_mode()
            else:
                logger.warning("Grasp failed after %d retries — giving up", GRASP_MAX_RETRIES)
                break

        logger.info("servo run finished")

    def _run(self):
        logger.info("servo worker ready")
        while not self.stop_event.is_set():
            if not self._run_event.wait(timeout=0.5):
                continue
            if self.stop_event.is_set():
                break
            try:
                self._execute_run()
            except Exception as e:
                logger.exception("servo worker crashed: %s", e)
                self.exc = e
            finally:
                self.cleanup()
                self._run_event.clear()
                if self._on_run_complete:
                    self._on_run_complete()
