import threading
import time
from typing import Any, Dict, Optional, Tuple


class StreamHub:
    """Thread-safe latest-frame store for all streaming channels."""

    _CHANNELS = {"servo", "inspect", "wrist_raw", "base_raw"}

    def __init__(self):
        self._lock = threading.Lock()
        self._seq: Dict[str, int] = {ch: 0 for ch in self._CHANNELS}
        self._latest: Dict[str, Optional[Dict[str, Any]]] = {ch: None for ch in self._CHANNELS}
        self._active_mode = "idle"

    def set_active_mode(self, mode: str):
        with self._lock:
            self._active_mode = mode

    def get_active_mode(self) -> str:
        with self._lock:
            return self._active_mode

    def publish(self, channel: str, payload: Dict[str, Any]):
        if channel not in self._latest:
            raise ValueError(f"Unknown channel: {channel}")
        with self._lock:
            self._seq[channel] += 1
            payload = dict(payload)
            payload["frame_id"] = self._seq[channel]
            payload["timestamp"] = time.time()
            self._latest[channel] = payload

    def snapshot(self, channel: str) -> Tuple[int, Optional[Dict[str, Any]]]:
        if channel not in self._latest:
            raise ValueError(f"Unknown channel: {channel}")
        with self._lock:
            if self._latest[channel] is None:
                return self._seq[channel], None
            return self._seq[channel], dict(self._latest[channel])
