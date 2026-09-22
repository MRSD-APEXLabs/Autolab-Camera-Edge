#!/usr/bin/env python3
"""ZED X capture process: opens the camera with the ZED SDK and writes JPEG pairs to stdout.

pyzed holds the GIL through `Camera.open()` (~5 s) and `close()` (~0.6 s), so inside the hub it
froze every request, /status and /time included, on each switch to the ZED X. The hub therefore
runs this script as a child (zedx_source.ZedXSession) and reads its records from a pipe. SIGTERM
closes the camera and exits; the hub kills the process only when the SDK hangs. Each record is
b'ZX', a kind byte and a uint32 body length, then the body:

    O  opened: JSON {"open_s"}
    F  pair: capture_ns, capture_mono_ns (int64), left JPEG length (uint32), left JPEG, right JPEG
    S  stats: JSON {"fps", "grab_fps", "encode_ms"}, every 2 s (also while grabs fail)
    E  error: UTF-8 text
    C  closed: the camera is released. The SDK's teardown at exit takes another ~0.6 s, which
       the hub does not wait for.

Only the frames that are streamed (`--stream-fps` of the 60 fps capture) are retrieved and
encoded. Both eyes come from one grab and share its capture stamp.
"""
import argparse
import json
import os
import signal
import struct
import sys
import time

import cv2

WIDTH, HEIGHT, CAPTURE_FPS = 960, 600, 60      # sl.RESOLUTION.SVGA
MAGIC = b'ZX'
HEADER = struct.Struct('<2scI')                # magic, kind, body length
PAIR = struct.Struct('<qqI')                   # capture_ns, capture_mono_ns, left JPEG length
OPENED, PAIR_KIND, STATS, ERROR, CLOSED = b'O', b'F', b'S', b'E', b'C'
MAX_RECORD = 16 << 20
MAX_STAMP_AGE_NS = 1_000_000_000               # an image stamp further from the wall clock is not believed
STATS_WINDOW_S = 2.0


def write_record(out, kind, *parts):
    out.write(HEADER.pack(MAGIC, kind, sum(len(part) for part in parts)))
    for part in parts:
        out.write(part)
    out.flush()


def capture(sl, serial, stream_fps, quality, grab_error_timeout, send, stopping):
    """Open the ZED X and send pairs until stopping() is true; returns an error text ('' after a stop)."""
    init = sl.InitParameters(camera_resolution=sl.RESOLUTION.SVGA, camera_fps=CAPTURE_FPS,
                             depth_mode=sl.DEPTH_MODE.NONE, sdk_verbose=0)
    init.set_from_serial_number(serial)
    zed = sl.Camera()
    t0 = time.monotonic()
    try:
        err = zed.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            return 'ZED X SN%d open failed: %s' % (serial, err)
        resolution = zed.get_camera_information().camera_configuration.resolution
        if (resolution.width, resolution.height) != (WIDTH, HEIGHT):
            # The /info geometry and the calibration section (SVGA) assume 960x600.
            return 'ZED X opened at %dx%d instead of %dx%d' % (resolution.width, resolution.height, WIDTH, HEIGHT)
        if stopping():
            return ''
        send(OPENED, json.dumps({'open_s': round(time.monotonic() - t0, 2)}).encode())
        return _grab_loop(sl, zed, stream_fps, quality, grab_error_timeout, send, stopping)
    finally:
        zed.close()


def _grab_loop(sl, zed, stream_fps, quality, grab_error_timeout, send, stopping):
    runtime = sl.RuntimeParameters(enable_depth=False)
    left, right = sl.Mat(), sl.Mat()
    params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    # Publish a frame once a stream period has passed, with half a capture period of slack for jitter.
    min_gap_ns = max(0, int(1e9 / stream_fps - 0.5e9 / CAPTURE_FPS))
    last_mono, age_ns, failing_since = None, 0, None
    t_win, n_grab, n_pub, encode_s = time.monotonic(), 0, 0, 0.0
    while not stopping():
        err = zed.grab(runtime)
        if err != sl.ERROR_CODE.SUCCESS:
            now = time.monotonic()
            failing_since = failing_since or now
            if now - failing_since > grab_error_timeout:
                return 'ZED X grab failing for %.1f s: %s' % (now - failing_since, err)
            time.sleep(0.01)
        else:
            failing_since = None
            n_grab += 1
            wall_ns, mono_ns = time.time_ns(), time.monotonic_ns()
            image_ns = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
            # The SDK stamps with the wall clock; its age maps the stamp to CLOCK_MONOTONIC. The last
            # plausible age is kept when the two disagree (the clock stepped since the open, e.g. NTP
            # at boot), so a clock step neither shifts the monotonic stamps nor stops the decimation.
            if 0 <= wall_ns - image_ns < MAX_STAMP_AGE_NS:
                age_ns = wall_ns - image_ns
            capture_mono_ns = mono_ns - age_ns
            if last_mono is None or not 0 <= capture_mono_ns - last_mono < min_gap_ns:
                last_mono = capture_mono_ns
                t0 = time.monotonic()
                zed.retrieve_image(left, sl.VIEW.LEFT_UNRECTIFIED)
                zed.retrieve_image(right, sl.VIEW.RIGHT_UNRECTIFIED)
                # BGRA goes straight to the encoder, which drops the alpha channel itself (saves a cvtColor).
                ok_left, left_jpg = cv2.imencode('.jpg', left.get_data(), params)
                ok_right, right_jpg = cv2.imencode('.jpg', right.get_data(), params)
                encode_s += time.monotonic() - t0
                if ok_left and ok_right:
                    left_bytes = left_jpg.tobytes()
                    send(PAIR_KIND, PAIR.pack(wall_ns - age_ns, capture_mono_ns, len(left_bytes)),
                         left_bytes, right_jpg.tobytes())
                    n_pub += 1
        elapsed = time.monotonic() - t_win
        if elapsed >= STATS_WINDOW_S:
            send(STATS, json.dumps({'fps': round(n_pub / elapsed, 1), 'grab_fps': round(n_grab / elapsed, 1),
                                    'encode_ms': round(1000 * encode_s / max(n_pub, 1), 1)}).encode())
            t_win, n_grab, n_pub, encode_s = time.monotonic(), 0, 0, 0.0
    return ''


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--serial', type=int, required=True)
    parser.add_argument('--stream-fps', type=float, default=30.0)
    parser.add_argument('--quality', type=int, default=80)
    parser.add_argument('--grab-error-timeout', type=float, default=3.0,
                        help='grab errors for this long end the process')
    parser.add_argument('--cv-threads', type=int, default=1)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    # Records go to the original stdout; anything the SDK prints lands on stderr instead.
    out = os.fdopen(os.dup(1), 'wb')
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    # A plain flag: setting an Event from a signal handler can deadlock on the Event's own lock.
    signals = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, frame: signals.append(signum))
    cv2.setNumThreads(max(0, args.cv_threads))

    def send(kind, *parts):
        write_record(out, kind, *parts)
    try:
        try:
            import pyzed.sl as sl
        except ImportError as exc:
            error = 'pyzed (ZED SDK Python API) is not installed: %s' % exc
        else:
            error = capture(sl, args.serial, args.stream_fps, args.quality, args.grab_error_timeout, send,
                            lambda: bool(signals))
    except BrokenPipeError:      # the hub went away; capture() has closed the camera
        return 0
    except Exception as exc:     # SDK or OpenCV failure: the hub reopens the camera
        error = 'ZED X capture error: %r' % exc
    try:
        if error:
            send(ERROR, error.encode())
        send(CLOSED)
        out.close()
    except OSError:
        pass
    return 1 if error else 0


if __name__ == '__main__':
    sys.exit(main())
