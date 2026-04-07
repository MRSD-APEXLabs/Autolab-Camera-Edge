import threading
from typing import Optional


class CameraWorker(threading.Thread):
    """Base class for single-camera processing threads.

    Subclasses receive an already-open camera and a grab lock.
    They must NOT open or close the camera.
    All camera.grab() calls must be made while holding grab_lock.
    """

    def __init__(self, name: str, camera, grab_lock: threading.Lock):
        super().__init__(daemon=True)
        self.name = name
        self.camera = camera
        self.grab_lock = grab_lock
        self.stop_event = threading.Event()
        self.exc: Optional[BaseException] = None

    def stop(self):
        self.stop_event.set()

    def should_stop(self) -> bool:
        return self.stop_event.is_set()

    def run(self):
        raise NotImplementedError
