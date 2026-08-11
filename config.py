import numpy as np


# =============================================================================
# Configuration
# =============================================================================

DEBUG = False

SERIAL_CAM1 = "40128964"  # visual servo camera
SERIAL_CAM2 = "42757821"  # pointcloud + apriltags + second YOLO camera
ARM_IP = "192.168.1.236"

MODEL_SERVO = "models/best_wrist.pt"
MODEL_INSPECT = "models/best_top.pt"   # set your second model path here

CONTROL_PORT = 8765
STREAM_PORT = 8766
RAW_STREAM_PORT = 8767

AREA_STOP_THRESHOLD = 50000.0

GRASP_MAX_RETRIES = 3
GRASP_RETRY_LIFT_MM = 50.0
# FSR_GRASP_THRESHOLD = 380             # FSR value (0–1023) above which grasp is considered successful
# FSR_BASELINE_SAMPLES = 5              # readings averaged for pre-close baseline
# FSR_CLOSE_NUDGE_RETRIES = 2          # re-close + nudge attempts if grasp threshold not met
# FSR_NUDGE_MIN_DELTA = 50              # |fsr1 - fsr2| below this → no nudge
# FSR_NUDGE_MM = 0.5                    # mm to nudge per correction
# FSR_FINGER_TOOL_AXIS = "y"            # tool-frame axis separating the two fingers
GRASP_DESCENT_MM = 70.0     # fallback fixed descent when OBB-based estimate is unavailable
GRASP_YAW_OFFSET_DEG = 0.0  # additional yaw correction for physical gripper calibration

# OBB-based depth estimation for adaptive descent.
# Z_mm = sqrt(fx * fy * PLATE_WIDTH_MM * PLATE_HEIGHT_MM / pixel_area)
# Set to the physical dimensions (mm) of what the servo YOLO model detects.
# Standard SBS 96-well plate outer footprint: 127.76 × 85.48 mm.
PLATE_WIDTH_MM = 127.76
PLATE_HEIGHT_MM = 85.48
# CAMERA_TO_GRASP_OFFSET_MM: distance (mm) from the ZED right-camera optical center to
# the gripper fingertip grasp plane along the camera Z-axis.  Subtract from the
# estimated depth to get the required arm descent.
CAMERA_TO_GRASP_OFFSET_MM = 115.0
START_POS_MM = np.array([-2.6, 320.0, 222.0], dtype=np.float64)
START_RPY_DEG = np.array([-180.0, 0.0, 0.0], dtype=np.float64)

POINTCLOUD_STRIDE = 8
STREAM_HZ = 15.0

# 60 fps is required to avoid an intermittent MAX96712 dual-link init race on the
# ZEDLinkDuo + Jetson Xavier stack. At 30 fps the GMSL2 link timing triggers a
# race condition in the sl_max96712 driver when the second camera registers while
# the first is already streaming. Do not lower this without re-validating startup.
CAMERA_FPS = 60


NUDGE_MM         = 1.0   # mm per nudge
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
