"""Pieces shared by the hub's cameras: frame slots, factory calibration and the camera base class.

A camera object lives as long as the hub, while its capture sessions come and go with every
switch. Everything a client can observe across a switch therefore lives here: the latest
frames (with frame ids that never go backwards), the calibration and the cumulative stats.
"""
import logging
import os
import threading
import time
import urllib.request

CALIBRATION_URL = 'https://calib.stereolabs.com/?SN=%d'
EYES = ('left', 'right')

log = logging.getLogger('camera_hub')


class FrameSlot:
    """Latest frame of one stream plus a condition to wait for the next one.

    Frame ids count for the life of the hub, across capture restarts and camera switches.
    `reset()` drops the frame and wakes every waiter. Deactivating the camera calls it, so its
    feeds notice at once instead of after their 1 s wait, and so does activating it, so a new
    activation never serves a frame of the last one.
    """

    def __init__(self):
        self.cond = threading.Condition()
        self.payload = None
        self.frame_id = 0
        self.meta = {'frame_id': 0, 'capture_ns': 0, 'ready_ns': 0, 'capture_mono_ns': 0, 'ready_mono_ns': 0}
        self.updated = 0.0      # time.monotonic() of the last publish
        self._resets = 0

    def publish(self, payload, capture_ns, capture_mono_ns):
        """Store a frame (wall-clock and CLOCK_MONOTONIC capture stamps in ns); returns its frame id."""
        with self.cond:
            self.frame_id += 1
            self.payload = payload
            self.meta = {'frame_id': self.frame_id, 'capture_ns': capture_ns, 'ready_ns': time.time_ns(),
                         'capture_mono_ns': capture_mono_ns, 'ready_mono_ns': time.monotonic_ns()}
            self.updated = time.monotonic()
            self.cond.notify_all()
            return self.frame_id

    def latest(self, last_id, timeout=1.0):
        """Block until a frame newer than last_id exists (or timeout, or reset); return (payload, meta)."""
        deadline = time.monotonic() + timeout
        with self.cond:
            resets = self._resets
            while (self.payload is None or self.frame_id == last_id) and self._resets == resets:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self.cond.wait(remaining)
            return self.payload, dict(self.meta)

    def reset(self):
        with self.cond:
            self.payload = None
            self._resets += 1
            self.cond.notify_all()


class Calibration:
    """Stereolabs .conf text for a serial: `path` if given, else the cached SN<serial>.conf in
    `cache_dir`, else downloaded once and cached; retried lazily on failure.

    `section` is the capture mode the hub streams (SVGA, FHD1200): a file without it is useless.
    """

    def __init__(self, serial, cache_dir, section, path=None, retry_interval=60.0):
        self.serial, self.cache_dir, self.section, self.path = serial, cache_dir, section, path
        self.retry_interval = retry_interval
        self._text = None
        self._next_try = 0.0
        self._lock = threading.Lock()

    @property
    def available(self):
        return self.text() is not None

    def text(self):
        with self._lock:
            if self._text is None and time.monotonic() >= self._next_try:
                self._next_try = time.monotonic() + self.retry_interval
                self._text = self._load()
            return self._text

    def _usable(self, text):
        return '[STEREO]' in text and '[LEFT_CAM_%s]' % self.section in text

    def _load(self):
        if self.path:
            try:
                with open(self.path) as f:
                    text = f.read()
                if self._usable(text):
                    return text
                log.warning('%s has no [STEREO]/[LEFT_CAM_%s] section; trying the cache', self.path, self.section)
            except OSError as exc:
                log.warning('cannot read calibration %s: %s; trying the cache', self.path, exc)
        if not self.serial:
            return None
        cache = os.path.join(self.cache_dir, 'SN%d.conf' % self.serial)
        if os.path.exists(cache):
            with open(cache) as f:
                text = f.read()
            if self._usable(text):
                return text
        try:
            text = urllib.request.urlopen(CALIBRATION_URL % self.serial, timeout=15).read().decode('utf-8', 'replace')
        except Exception as exc:
            log.warning('calibration download failed for SN%d: %s', self.serial, exc)
            return None
        if not self._usable(text):
            log.warning('calibration server returned no usable file for SN%d', self.serial)
            return None
        try:
            tmp = cache + '.tmp'
            with open(tmp, 'w') as f:
                f.write(text)
            os.replace(tmp, cache)
        except OSError as exc:
            log.warning('could not cache calibration: %s', exc)
        return text


class CameraSource:
    """A camera the hub can switch to.

    Every activation opens a new capture session with `open_session()`: an object with
    start(), stop(timeout) -> released, `alive` (False once it gave up) and `error` ('' while
    healthy). The hub stops it when another camera is selected and replaces it when it fails.
    Subclasses provide the slots, `open_session()`, `info()` (the /info schema of the old Nano
    server), `latest()`, `stereo_pairs()` and `stats()`.
    """
    backend = ''

    def __init__(self, name, model, serial, role, calibration, slots):
        self.name, self.model, self.serial, self.role = name, model, int(serial), role
        self.calibration = calibration
        self.slots = slots
        self.active = False       # selected in the hub
        self.activation = 0       # counts selections; a feed ends when the one it started in is over
        self.restarts = 0         # failed sessions the hub replaced
        self.last_error = ''

    def activate(self):
        for slot in self.slots:
            slot.reset()          # a frame the last session published while it was being stopped is stale now
        self.activation += 1
        self.active = True

    def deactivate(self):
        self.active = False
        for slot in self.slots:
            slot.reset()

    def is_live(self, activation):
        return self.active and self.activation == activation

    def record_failure(self, error):
        self.restarts += 1
        self.last_error = error[-300:]

    def frames_since(self, since):
        """True when every slot got a frame at or after monotonic time `since`."""
        return all(slot.payload is not None and slot.updated >= since for slot in self.slots)

    def frame_age(self):
        """Seconds since the stalest slot got a frame (inf if one never did)."""
        now = time.monotonic()
        return max(now - slot.updated if slot.payload is not None else float('inf') for slot in self.slots)

    def frames(self, eye, alive):
        """Yield (jpeg, meta) for every new frame of one eye, None when idle for ~1 s; ends when alive() is False."""
        last = 0
        while alive():
            jpg, meta = self.latest(eye, last, timeout=1.0)
            if jpg is None or meta['frame_id'] == last:
                yield None
                continue
            last = meta['frame_id']
            yield jpg, meta

    def status(self):
        status = {'model': self.model, 'serial': self.serial, 'role': self.role, 'backend': self.backend,
                  'active': self.active, 'activation': self.activation, 'restarts': self.restarts,
                  'last_error': self.last_error}
        status.update(self.stats())
        return status

    # -- subclass API -----------------------------------------------------------------------------
    def open_session(self):
        raise NotImplementedError

    def info(self):
        raise NotImplementedError

    def latest(self, eye, last_id, timeout=1.0):
        raise NotImplementedError

    def stereo_pairs(self, alive):
        """Yield (left_jpeg, left_meta, right_jpeg, right_meta) per synchronized pair, None when idle for ~1 s."""
        raise NotImplementedError

    def stats(self):
        return {}
