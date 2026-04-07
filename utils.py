import asyncio
import json
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import zlib
import base64
import zlib
import subprocess

import cv2
import numpy as np
import pyzed.sl as sl
import websockets
from ultralytics import YOLO
from xarm.wrapper import XArmAPI
import apriltag
from config import *

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


def draw_velocity_arrow(
    img, center_uv, Vc, scale=600.0, color=(0, 0, 255), thickness=3
):
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
    cls_id = (
        int(r0.obb.cls.detach().cpu().numpy()[idx]) if r0.obb.cls is not None else -1
    )
    conf = float(confs[idx])

    u = float(xywhr[0])
    v = float(xywhr[1])
    w = float(xywhr[2])
    h = float(xywhr[3])
    theta = float(xywhr[4])

    return u, v, w, h, theta, conf, cls_id


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

    Vc = np.array(
        [
            kx * ex,
            -ky * ey,
            0.25 * ez,
            0.0,
            0.0,
            -kw * etheta,
        ],
        dtype=np.float64,
    )

    return Vc, (u, v, w, h, theta, conf, cls_id)


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


def encode_jpeg_b64(frame: np.ndarray, quality: int = 80) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("Failed to JPEG-encode frame")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def encode_zlib_b64(arr: np.ndarray) -> Tuple[str, List[int], str]:
    arr = np.asarray(arr)
    payload = zlib.compress(arr.tobytes(), level=6)
    return base64.b64encode(payload).decode("ascii"), list(arr.shape), str(arr.dtype)


def all_obb_detections(results):
    """Return all OBB detections as JSON-friendly dictionaries."""
    r0 = results[0]
    if r0.obb is None or len(r0.obb) == 0:
        return []

    xywhr = r0.obb.xywhr.detach().cpu().numpy()
    confs = r0.obb.conf.detach().cpu().numpy()
    cls_ids = (
        r0.obb.cls.detach().cpu().numpy()
        if r0.obb.cls is not None
        else np.full(len(confs), -1)
    )
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
