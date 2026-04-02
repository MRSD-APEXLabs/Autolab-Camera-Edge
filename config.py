import numpy as np

# =============================================================================
# Configuration
# =============================================================================

DEBUG = True

SERIAL_CAM1 = "40128964"   # visual servo camera
SERIAL_CAM2 = "42757821"   # pointcloud + apriltags + second YOLO camera
ARM_IP = "192.168.1.236"

MODEL_SERVO = "models/best_wrist.pt"
MODEL_INSPECT = "models/best_top.pt"

CONTROL_PORT = 8765
STREAM_PORT = 8766

AREA_STOP_THRESHOLD = 50000.0
START_POS_MM = np.array([-2.6, 320.0, 222.0], dtype=np.float64)
START_RPY_DEG = np.array([-180.0, 0.0, 0.0], dtype=np.float64)

POINTCLOUD_STRIDE = 8   # larger = smaller websocket payload
STREAM_HZ = 2.0