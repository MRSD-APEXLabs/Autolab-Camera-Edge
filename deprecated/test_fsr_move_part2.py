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
from utils import setup_gripper, gripper_get_fsr, gripper_open
from config import ARM_IP

# --- tunables ---
NUDGE_MM         = 5.0   # mm per nudge
FINGER_TOOL_AXIS = "y"   # tool-frame axis that separates the two fingers ("x" or "y")
N_SAMPLES        = 10    # total FSR readings
LOOP_HZ          = 2     # readings per second
FSR_MIN_DELTA    = 50    # skip move if |fsr1 - fsr2| < this
N_SAMPLES2       = 500
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
    gripper_open(arm)
    print(f"Ready. FINGER_TOOL_AXIS={FINGER_TOOL_AXIS}, NUDGE_MM={NUDGE_MM}\n")

    try:
        fsr1_values = []
        fsr2_values = []
        for i in range(N_SAMPLES):
            fsr1, fsr2 = gripper_get_fsr(arm)
            fsr1_values.append(fsr1)
            fsr2_values.append(fsr2)
        
        fsr1_avg = np.mean([v for v in fsr1_values if v is not None])
        fsr2_avg = np.mean([v for v in fsr2_values if v is not None])
        fsr1_avg = fsr1_avg -20 if fsr1_avg > 20 else fsr1_avg
        fsr2_avg = fsr2_avg -20 if fsr2_avg > 20 else fsr2_avg
        print(f"Average FSR1: {fsr1_avg:.2f}, Average FSR2: {fsr2_avg:.2f}")
        
        for i in range(N_SAMPLES2):
            fsr1, fsr2 = gripper_get_fsr(arm)
            # take logarithmic
            print(f"Raw FSR1={fsr1}, FSR2={fsr2}")
            print(f"Differece: {fsr1-fsr1_avg:.2f}, {fsr2-fsr2_avg:.2f}")
            actual_fsr1 = np.log(fsr1-fsr1_avg + 1) if fsr1 is not None else None
            actual_fsr2 = np.log(fsr2-fsr2_avg + 1) if fsr2 is not None else None
            
            print(f"  [{i+1}/{N_SAMPLES2}] FSR1={fsr1:4d} (log={actual_fsr1:.2f})  FSR2={fsr2:4d} (log={actual_fsr2:.2f})")
            
            print(f"Delta:", abs(actual_fsr1 - actual_fsr2) if actual_fsr1 is not None and actual_fsr2 is not None else "N/A")
            # if abs(actual_fsr1 - actual_fsr2) < FSR_MIN_DELTA:
            #     print("           balanced / below threshold — no move")
            # elif actual_fsr1 > actual_fsr2:
            #     print(f"           FSR1 dominant → -{NUDGE_MM}mm along tool-{FINGER_TOOL_AXIS}")
            #     nudge(arm, -NUDGE_MM)
            # else:
            #     print(f"           FSR2 dominant → +{NUDGE_MM}mm along tool-{FINGER_TOOL_AXIS}")
            #     nudge(arm, +NUDGE_MM)
            time.sleep(1.0 / LOOP_HZ)
            
        # for i in range(N_SAMPLES):
        #     fsr1, fsr2 = gripper_get_fsr(arm)
        #     if fsr1 is None:
        #         print(f"  [{i+1}/{N_SAMPLES}] FSR read failed")
        #         time.sleep(1.0 / LOOP_HZ)
        #         continue

        #     delta = fsr1 - fsr2
        #     print(f"  [{i+1}/{N_SAMPLES}] FSR1={fsr1:4d}  FSR2={fsr2:4d}  delta={delta:+d}")

        #     if abs(delta) < FSR_MIN_DELTA:
        #         print("           balanced / below threshold — no move")
        #     elif delta > 0:
        #         print(f"           FSR1 dominant → -{NUDGE_MM}mm along tool-{FINGER_TOOL_AXIS}")
        #         nudge(arm, -NUDGE_MM)
        #     else:
        #         print(f"           FSR2 dominant → +{NUDGE_MM}mm along tool-{FINGER_TOOL_AXIS}")
        #         nudge(arm, +NUDGE_MM)

        #     time.sleep(1.0 / LOOP_HZ)

    except KeyboardInterrupt:
        print("\nAborted.")
    finally:
        arm.set_mode(1)
        arm.set_state(0)
        arm.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
