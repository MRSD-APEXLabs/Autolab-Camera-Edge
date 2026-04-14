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
GRASP_DESCENT_MM = 50.0     # mm to descend below IBVS-converged Z to reach grasp height
GRASP_YAW_OFFSET_DEG = 0.0  # additional yaw correction for physical gripper calibration
START_POS_MM = np.array([-2.6, 320.0, 222.0], dtype=np.float64)
START_RPY_DEG = np.array([-180.0, 0.0, 0.0], dtype=np.float64)

POINTCLOUD_STRIDE = 8
STREAM_HZ = 15.0

# 60 fps is required to avoid an intermittent MAX96712 dual-link init race on the
# ZEDLinkDuo + Jetson Xavier stack. At 30 fps the GMSL2 link timing triggers a
# race condition in the sl_max96712 driver when the second camera registers while
# the first is already streaming. Do not lower this without re-validating startup.
CAMERA_FPS = 60