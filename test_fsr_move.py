#!/usr/bin/env python3
"""
Test script: read FSR1/FSR2, nudge end-effector toward higher-pressure finger.

FSR1 (0x0702) and FSR2 (0x0703) are on opposing gripper fingers.
FINGER_TOOL_AXIS is the finger-separation axis in the *tool* frame.
The world-frame delta is computed by rotating that axis by current yaw,
so the nudge direction is always correct regardless of arm orientation.
"""

import time
import numpy as np

from xarm.wrapper import XArmAPI
from utils import setup_gripper, gripper_get_fsr
from config import ARM_IP

# --- tunables ---
NUDGE_MM         = 5.0   # mm per nudge
FINGER_TOOL_AXIS = "x"   # tool-frame axis that separates the two fingers ("x" or "y")
N_SAMPLES        = 10    # total FSR readings
LOOP_HZ          = 2     # readings per second
FSR_MIN_DELTA    = 50    # skip move if |fsr1 - fsr2| < this
# ----------------


def connect_arm():
    arm = XArmAPI(ARM_IP)
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)
    return arm


def get_pose(arm):
    ret = arm.get_position(is_radian=False)
    if not isinstance(ret, tuple) or len(ret) < 2 or ret[0] != 0:
        raise RuntimeError(f"get_position failed: {ret}")
    return ret[1][:6]  # x, y, z, roll, pitch, yaw (degrees)


def tool_axis_in_world(yaw_deg, axis):
    """Rotate a unit tool-frame axis vector into world frame using yaw."""
    yaw = np.radians(yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    if axis == "x":
        return np.array([c, s, 0.0])
    if axis == "y":
        return np.array([-s, c, 0.0])
    if axis == "z":
        return np.array([0.0, 0.0, 1.0])
    raise ValueError(f"Unknown axis: {axis}")


def nudge(arm, signed_mm):
    x, y, z, roll, pitch, yaw = get_pose(arm)
    direction = tool_axis_in_world(yaw, FINGER_TOOL_AXIS)
    delta = direction * signed_mm
    print(f"           yaw={yaw:.1f}° → world delta: dx={delta[0]:+.2f} dy={delta[1]:+.2f} dz={delta[2]:+.2f}")

    code = arm.set_position(
        x=x + delta[0], y=y + delta[1], z=z + delta[2],
        roll=roll, pitch=pitch, yaw=yaw,
        speed=30, wait=True,
    )
    if code != 0:
        print(f"           move failed: code={code}")


def main():
    print(f"Connecting to arm at {ARM_IP}...")
    arm = connect_arm()
    setup_gripper(arm)
    print(f"Ready. FINGER_TOOL_AXIS={FINGER_TOOL_AXIS}, NUDGE_MM={NUDGE_MM}\n")

    try:
        for i in range(N_SAMPLES):
            fsr1, fsr2 = gripper_get_fsr(arm)
            if fsr1 is None:
                print(f"  [{i+1}/{N_SAMPLES}] FSR read failed")
                time.sleep(1.0 / LOOP_HZ)
                continue

            delta = fsr1 - fsr2
            print(f"  [{i+1}/{N_SAMPLES}] FSR1={fsr1:4d}  FSR2={fsr2:4d}  delta={delta:+d}")

            if abs(delta) < FSR_MIN_DELTA:
                print("           balanced / below threshold — no move")
            elif delta > 0:
                print(f"           FSR1 dominant → +{NUDGE_MM}mm along tool-{FINGER_TOOL_AXIS}")
                nudge(arm, +NUDGE_MM)
            else:
                print(f"           FSR2 dominant → -{NUDGE_MM}mm along tool-{FINGER_TOOL_AXIS}")
                nudge(arm, -NUDGE_MM)

            time.sleep(1.0 / LOOP_HZ)

    except KeyboardInterrupt:
        print("\nAborted.")
    finally:
        arm.set_mode(1)
        arm.set_state(0)
        arm.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
