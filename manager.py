"""Selection and supervision of the hub's cameras: at most one streams at a time.

The ZED X and the ZED X Nano sit on the same GMSL deserializer and can run together, but the
Xavier's CPU is better spent on one camera and the Thor needs only one at a time. Selecting a
camera deactivates the current one (its feeds end), releases its devices, persists the choice
in state.json and starts the new one. A supervisor thread tracks the active camera:
`streaming` while every eye delivers frames, `error` when the capture fails or stalls. A
session that gave up is reopened with a backoff for as long as the camera stays selected.
"""
import json
import logging
import os
import threading
import time

HUB_VERSION = 1
IDLE, STARTING, STREAMING, ERROR = 'idle', 'starting', 'streaming', 'error'
STALL_S = 3.0                      # an eye without frames for this long puts the camera in the error state
START_TIMEOUT_S = 20.0             # no frames this long after a start -> error (the session keeps trying)
RETRY_DELAYS_S = (2.0, 3.0, 5.0)   # backoff before reopening a failed session
STOP_TIMEOUT_S = 15.0              # the ZED SDK cannot abort an open() in progress (~6 s)

log = logging.getLogger('camera_hub')


class CameraHub:
    """Owns the cameras and the one active capture session.

    `select()` and the supervisor's restarts are serialized by one lock; status reads take only
    the condition that guards the state fields, so they never wait for a camera to open or close.
    """

    def __init__(self, cameras, state_path, default=None, stall_s=STALL_S, start_timeout_s=START_TIMEOUT_S,
                 retry_delays_s=RETRY_DELAYS_S, stop_timeout_s=STOP_TIMEOUT_S, tick_s=0.1):
        self.cameras = {camera.name: camera for camera in cameras}
        self.state_path = state_path
        self.stall_s, self.start_timeout_s = stall_s, start_timeout_s
        self.retry_delays_s, self.stop_timeout_s, self.tick_s = retry_delays_s, stop_timeout_s, tick_s
        self._switch_lock = threading.Lock()   # session lifecycle: select() and the supervisor
        self._cond = threading.Condition()     # guards the fields below; notified on every state change
        self.active = None
        self.state, self.error, self._state_since = IDLE, None, time.monotonic()
        self.switch_count = 0
        self._generation = 0                   # counts activations, so a waiting select() notices a newer one
        self._session = None
        self._session_started = 0.0
        self._failures = 0
        self._restart_at = None
        self._started = time.monotonic()
        self._started_ns = time.time_ns()      # tells clients that the hub restarted (their feeds ended)
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._supervise, name='hub-supervisor', daemon=True)
        self._initial = self._load_state(default)

    # -- persistence ------------------------------------------------------------------------------
    def _load_state(self, default):
        try:
            with open(self.state_path) as f:
                active = json.load(f)['active']
        except FileNotFoundError:
            return default
        except (OSError, ValueError, KeyError, TypeError) as exc:
            log.warning('ignoring unreadable %s (%s); using %s', self.state_path, exc, default)
            return default
        if active is not None and active not in self.cameras:
            log.warning('%s selects unknown camera %r; using %s', self.state_path, active, default)
            return default
        return active

    def _save_state(self):
        try:
            tmp = self.state_path + '.tmp'
            with open(tmp, 'w') as f:
                json.dump({'active': self.active}, f)
            os.replace(tmp, self.state_path)
        except OSError as exc:
            log.warning('could not save the selection to %s: %s', self.state_path, exc)

    # -- lifecycle --------------------------------------------------------------------------------
    def start(self):
        """Activate the persisted (or default) camera and start supervising; does not wait for frames."""
        with self._switch_lock:
            self._activate(self._initial, user=False)
        self._thread.start()

    def close(self):
        """Release the active camera (the selection stays persisted for the next start)."""
        self._closed.set()
        with self._switch_lock:
            if self.active is not None:
                self.cameras[self.active].deactivate()
            self._stop_session()
            self._set_state(IDLE, None)
        if self._thread.is_alive():
            self._thread.join(1.0)

    # -- selection --------------------------------------------------------------------------------
    def select(self, name, wait=True, timeout=30.0):
        """Make `name` (None = no camera) the active camera.

        Returns 'ok', 'error' (start failed; the camera stays selected and retrying), 'timeout',
        'superseded' (another selection came in while waiting) or 'closed' (the hub is shutting
        down and opens nothing). Unknown names raise KeyError.
        """
        if name is not None and name not in self.cameras:
            raise KeyError(name)
        with self._switch_lock:
            if self._closed.is_set():
                return 'closed'   # a camera opened now would outlive the hub's close()
            with self._cond:
                healthy = self.state in (STARTING, STREAMING)
            if name != self.active or not healthy:
                self._activate(name, user=True)
            generation = self._generation
        if name is None or not wait:
            return 'ok'
        return self._wait_ready(generation, timeout)

    def _activate(self, name, user):
        """Switch to `name`, or restart it when it is already selected. Caller holds the switch lock."""
        previous = self.active
        with self._cond:
            self._generation += 1
            if previous != name:
                if previous is not None:
                    self.cameras[previous].deactivate()   # its feeds end now, before the slow release
                if name is not None:
                    self.cameras[name].activate()         # its feeds may connect and wait for the first frame
                self.active = name
                if user:
                    self.switch_count += 1
        self._restart_at = None
        self._set_state(IDLE if name is None else STARTING, None)
        t0 = time.monotonic()
        self._stop_session()
        if user and previous != name:
            self._save_state()
        if previous != name:
            log.info('selected %s (was %s; released in %.1f s)', name, previous, time.monotonic() - t0)
        if name is not None:
            self._failures = 0
            self._start_session(self.cameras[name])

    def _start_session(self, camera):
        self._session_started = time.monotonic()
        session = None
        try:
            session = camera.open_session()
            session.start()
        except Exception as exc:
            if session is not None:
                session.stop(self.stop_timeout_s)
            self._session_failed(camera, 'start failed: %r' % exc)
            return
        self._session = session

    def _session_failed(self, camera, error):
        """Report the error and schedule the reopen; the camera stays selected."""
        delay = self.retry_delays_s[min(self._failures, len(self.retry_delays_s) - 1)]
        self._failures += 1
        camera.record_failure(error)
        self._restart_at = time.monotonic() + delay
        self._set_state(ERROR, error)

    def _stop_session(self):
        session, self._session = self._session, None
        if session is not None and not session.stop(self.stop_timeout_s):
            log.warning('capture session did not stop within %.0f s; continuing without it', self.stop_timeout_s)

    def _wait_ready(self, generation, timeout):
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                if self._closed.is_set():
                    return 'closed'
                if self._generation != generation:
                    return 'superseded'
                if self.state == STREAMING:
                    return 'ok'
                if self.state == ERROR:
                    return 'error'
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return 'timeout'
                self._cond.wait(min(remaining, 0.5))

    # -- supervision ------------------------------------------------------------------------------
    def _set_state(self, state, error):
        with self._cond:
            if state != self.state:
                elapsed = time.monotonic() - self._state_since
                self._state_since = time.monotonic()
                if state == STREAMING or error:
                    log.info('%s: %s -> %s after %.1f s%s', self.active, self.state, state, elapsed,
                             ': ' + error if error else '')
            self.state, self.error = state, error
            self._cond.notify_all()

    def _supervise(self):
        while not self._closed.wait(self.tick_s):
            if not self._switch_lock.acquire(blocking=False):
                continue   # a selection is in progress
            try:
                self._tick()
            except Exception:
                log.exception('supervisor tick failed')
            finally:
                self._switch_lock.release()

    def _tick(self):
        if self.active is None or self._closed.is_set():
            return
        camera = self.cameras[self.active]
        now = time.monotonic()
        session = self._session
        if session is None:
            if self._restart_at is not None and now >= self._restart_at:
                self._restart_at = None
                log.info('reopening %s', camera.name)
                self._start_session(camera)
            return
        error = session.error
        if not session.alive:
            self._session = None
            session.stop(self.stop_timeout_s)
            self._session_failed(camera, error or 'capture stopped')
            return
        if error:
            self._set_state(ERROR, error)
        elif camera.frames_since(self._session_started) and camera.frame_age() < self.stall_s:
            self._failures = 0
            self._set_state(STREAMING, None)
        elif self.state == STREAMING:
            self._set_state(ERROR, 'no frames for %.1f s' % min(camera.frame_age(), now - self._session_started))
        elif self.state == STARTING and now - self._session_started > self.start_timeout_s:
            self._set_state(ERROR, 'no frames %g s after the start' % self.start_timeout_s)

    # -- status -----------------------------------------------------------------------------------
    def status(self):
        with self._cond:
            active, state, error, since = self.active, self.state, self.error, self._state_since
            switch_count, restart_at = self.switch_count, self._restart_at
        now = time.monotonic()
        return {
            'server': 'camera_hub', 'hub_version': HUB_VERSION, 'active': active, 'state': state, 'error': error,
            'state_since_s': round(now - since, 3), 'switch_count': switch_count,
            'retry_in_s': None if restart_at is None else round(max(0.0, restart_at - now), 1),
            'uptime_s': round(now - self._started, 1), 'started_ns': self._started_ns,
            'cameras': {name: camera.status() for name, camera in self.cameras.items()},
            't_ns': time.time_ns(), 'monotonic_ns': time.monotonic_ns(),
        }
