#!/usr/bin/env python3
"""
Test script: read FSR1/FSR2, nudge end-effector toward the lower-pressure finger.

FSR1 (0x0702) and FSR2 (0x0703) are on opposing gripper fingers.
If one sensor reaches the target pressure first, nudge toward the other side.
If both sensors exceed STOP_PRESSURE, stop and close the gripper.
"""

import time
import numpy as np

from xarm.wrapper import XArmAPI
from utils import setup_gripper, gripper_get_fsr, gripper_open, gripper_close, clear_errors, gripper_set
from config import ARM_IPe

# --- tunables ---
NUDGE_MM         = 5.0   # mm per nudge
FINGER_TOOL_AXIS  = "y"   # tool-frame axis that separates the two fingers ("x" or "y")
N_SAMPLES         = 100    # total FSR readings
LOOP_HZ           = 2     # readings per second

TARGET_PRESSURE_1   = 250   # if one finger reaches this, nudge
TARGET_PRESSURE_2   = 120   # if one finger reaches this, nudge
STOP_PRESSURE_1     = 370   # if both exceed this, stop
STOP_PRESSURE_2     = 140   # if both exceed this, stop
TARGET_TOLERANCE  = 20    # "around 400" window
GRIPPER_CLOSE = 50
GRIPPER_STEP = 20
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
    clear_errors(arm)
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)
    
    x, y, z, roll, pitch, yaw = get_pose(arm)
    direction = tool_axis_in_world(yaw, FINGER_TOOL_AXIS)
    delta = direction * signed_mm
    print(
        f"           yaw={yaw:.1f}° → world delta: "
        f"dx={delta[0]:+.2f} dy={delta[1]:+.2f} dz={delta[2]:+.2f}"
    )

    print("state:", arm.get_state(), "err:", arm.get_err_warn_code())
    code = arm.set_position(
        x=x + delta[0], y=y + delta[1], z=z + delta[2],
        roll=roll, pitch=pitch, yaw=yaw,
        speed=30, wait=True,
    )
    print(f"           move command result: code={code}")
    if code != 0:
        print(f"           move failed: code={code}")

def main():
    print(f"Connecting to arm at {ARM_IP}...")
    arm = connect_arm()
    setup_gripper(arm)
    gripper_open(arm)
    print(f"Ready. FINGER_TOOL_AXIS={FINGER_TOOL_AXIS}, NUDGE_MM={NUDGE_MM}\n")
    time.sleep(10.0)
    gripper_set(arm, GRIPPER_CLOSE)
    gripper_val = GRIPPER_CLOSE

    try:
        for i in range(N_SAMPLES):
            clear_errors(arm)
            fsr1, fsr2 = gripper_get_fsr(arm)
            if fsr1 is None:
                print(f"  [{i+1}/{N_SAMPLES}] FSR read failed")
                time.sleep(1.0 / LOOP_HZ)
                continue
            

            

            print(f"  [{i+1}/{N_SAMPLES}] FSR1={fsr1:4d}  FSR2={fsr2:4d}")

            # Stop condition: both fingers are pressing hard enough
            if fsr1 > STOP_PRESSURE_1 and fsr2 > STOP_PRESSURE_2:
                print("           both FSRs > 500 — stopping and closing gripper")
                gripper_set(arm, gripper_val+30)
                # gripper_close(arm)
                break

            # Nudge condition: one side is around 400 and the other is not
            fsr1_hit = (fsr1 - TARGET_PRESSURE_1 > 0)
            fsr2_hit = (fsr2 - TARGET_PRESSURE_2 > 0)

            if fsr1_hit and not fsr2_hit:
                print(f"           FSR1 near {TARGET_PRESSURE_1} and FSR2 not — nudge toward FSR2")
                nudge(arm, -NUDGE_MM)
            elif fsr2_hit and not fsr1_hit:
                print(f"           FSR2 near {TARGET_PRESSURE_2} and FSR1 not — nudge toward FSR1")
                nudge(arm, +NUDGE_MM)
            else:
                print("           no action")
                gripper_val += GRIPPER_STEP
                if gripper_val > 350:
                    gripper_val = 350
                gripper_set(arm, gripper_val)

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