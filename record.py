import os
import time
import threading
import pyzed.sl as sl


SERIAL_CAM1 = "42757821"
CAMERA_FPS = 60


class WristCameraRecorder:
    def __init__(self):
        self._cam = sl.Camera()
        self._lock = threading.Lock()

        init = sl.InitParameters()
        init.set_from_serial_number(SERIAL_CAM1)
        init.camera_resolution = sl.RESOLUTION.SVGA
        init.camera_fps = CAMERA_FPS
        init.depth_mode = sl.DEPTH_MODE.NONE

        err = self._cam.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open camera: {err}")

        self._recording = False
        self._thread = None


    def start(self, filename=None):
        if self._recording:
            return

        if filename is None:
            filename = f"wrist_{time.strftime('%Y%m%d_%H%M%S')}.svo2"

        os.makedirs("recordings", exist_ok=True)
        filepath = os.path.join("recordings", filename)

        rec_params = sl.RecordingParameters(
            filepath,
            sl.SVO_COMPRESSION_MODE.H264
        )

        with self._lock:
            err = self._cam.enable_recording(rec_params)

        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Recording start failed: {err}")

        self._recording = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

        print("Recording:", filepath)

    def _loop(self):
        runtime = sl.RuntimeParameters()

        while self._recording:
            with self._lock:
                self._cam.grab(runtime)

    def stop(self):
        if not self._recording:
            return

        self._recording = False
        self._thread.join()

        with self._lock:
            self._cam.disable_recording()

        print("Recording stopped")

    def close(self):
        self.stop()
        with self._lock:
            self._cam.close()



if __name__ == "__main__":
    cam = WristCameraRecorder()

    cam.start()
    time.sleep(100)
    cam.stop()

    cam.close()
