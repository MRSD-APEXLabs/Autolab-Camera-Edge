#!/usr/bin/env python3
"""Camera hub for the Jetson AGX Xavier: ZED X (base) or ZED X Nano (wrist) as stereo MJPEG, one at a time.

The ZED X opens through the ZED SDK (pyzed, in a child process: zedx_capture.py); the Nano needs
SDK >= 5.3, which JetPack 5 lacks, so it is read as raw V4L2 Bayer. Both are served as unrectified 960x600 JPEG pairs with the
wire format of the old Nano server (scripts/zedx_nano_v4l2_stream.py), so the Thor rectifies
and computes depth the same way for either camera. Switch with POST /select or on the page.

Run on the Xavier (Python 3.8, numpy, OpenCV, v4l2-ctl; pyzed for the ZED X):

    python3 camera_hub.py --port 8090      # then open http://<xavier-ip>:8090/

Global:      /                  page: active camera, switch buttons, live preview, stats
             /status            JSON: active camera, state (idle|starting|streaming|error), per-camera stats
             POST /select       {"camera": "zedx"|"zedx_nano"|null}; ?wait=0 returns at once, ?timeout=30
             /time              {"t_ns", "monotonic_ns"} for clock sync (shared by the cameras)
Per camera:  /cameras/<name>/info, /calibration.conf, /stats, /video_feed/<left|right|stereo>,
             /snapshot/<left|right>.jpg. Feeds and snapshots of the inactive camera answer 503.
Legacy:      /info, /calibration.conf, /video_feed[/<left|right|stereo>], /snapshot.jpg,
             /snapshot/<eye>.jpg, /stats of the old Nano server; they serve the Nano only.
"""
import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

from manager import HUB_VERSION, CameraHub
from nano_source import BAYER_CODES, NanoCamera
from sources import EYES, Calibration
from zedx_source import ZedXCamera

HUB_DIR = os.path.dirname(os.path.abspath(__file__))
LEGACY_CAMERA = 'zedx_nano'      # the old server's routes serve the wrist camera only
FEED_EYES = EYES + ('stereo',)
MULTIPART = 'multipart/x-mixed-replace; boundary=frame'

log = logging.getLogger('camera_hub')


def frame_headers(meta, sensor):
    return {'X-Frame-Id': str(meta['frame_id']), 'X-Capture-Ns': str(meta['capture_ns']),
            'X-Ready-Ns': str(meta['ready_ns']), 'X-Capture-Mono-Ns': str(meta['capture_mono_ns']),
            'X-Ready-Mono-Ns': str(meta['ready_mono_ns']), 'X-Sensor': sensor}


def _part(content_type, body_length, headers):
    head = '--frame\r\nContent-Type: %s\r\nContent-Length: %d\r\n' % (content_type, body_length)
    return (head + ''.join('%s: %s\r\n' % kv for kv in headers.items()) + '\r\n').encode('ascii')


def jpeg_part(jpg, meta, sensor, extra):
    headers = frame_headers(meta, sensor)
    headers.update(extra)
    return _part('image/jpeg', len(jpg), headers) + jpg + b'\r\n'


def stereo_part(ljpg, lmeta, rjpg, rmeta, extra):
    """One /video_feed/stereo part: left JPEG followed by the right one, as the old server sent it."""
    headers = frame_headers(lmeta, 'stereo')
    headers.update({'X-Left-Length': str(len(ljpg)), 'X-Right-Frame-Id': str(rmeta['frame_id']),
                    'X-Right-Capture-Ns': str(rmeta['capture_ns']),
                    'X-Sync-Us': str((lmeta['capture_mono_ns'] - rmeta['capture_mono_ns']) // 1000)})
    headers.update(extra)
    return _part('application/octet-stream', len(ljpg) + len(rjpg), headers) + ljpg + rjpg + b'\r\n'


PAGE = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>Camera hub</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{font-family:sans-serif;background:#111;color:#ddd;margin:16px}button{font-size:15px;padding:6px 12px;
margin-right:6px;background:#333;color:#ddd;border:1px solid #555;border-radius:4px;cursor:pointer}
button.on{background:#2a6;color:#fff;border-color:#2a6}button:disabled{opacity:.5;cursor:wait}
.err{color:#f66}.views{display:flex;flex-wrap:wrap;gap:8px}figure{margin:0;flex:1 1 400px;max-width:960px}
img{width:100%;border:1px solid #444;background:#000;min-height:60px}figcaption{color:#999}
pre{background:#222;padding:8px;overflow:auto}a{color:#8cf}</style></head><body>
<h2>Camera hub</h2>
<p>Active: <b id="active">?</b> &middot; state <b id="state">?</b> <span id="error" class="err"></span></p>
<p id="buttons"></p><p id="msg"></p>
<div class="views"><figure><img id="left" alt=""><figcaption>left</figcaption></figure>
<figure><img id="right" alt=""><figcaption>right</figcaption></figure></div>
<p><a href="/status">status</a> &middot; <a href="/time">time</a> &middot; per camera:
<code>/cameras/&lt;name&gt;/{info,calibration.conf,stats,video_feed/stereo}</code></p>
<pre id="stats">loading...</pre>
<script>
var shown = null, busy = false;
function $(id) { return document.getElementById(id); }
function show(s) {
  // Reconnect after a switch, a recovery, a release and reselect or a hub restart: the last two end
  // the open feeds and may both happen between two polls, so the key has the activation count and
  // the hub's start stamp.
  var c = s.active && s.cameras[s.active];
  var key = c && s.state === 'streaming' ? [s.active, c.activation, s.started_ns].join(':') : null;
  if (key === shown) return;
  shown = key;
  ['left', 'right'].forEach(function (eye) {
    if (key) $(eye).src = '/cameras/' + s.active + '/video_feed/' + eye + '?t=' + Date.now();
    else $(eye).removeAttribute('src');
  });
}
['left', 'right'].forEach(function (eye) {   // a feed that failed to open is retried at the next poll
  $(eye).onerror = function () { if ($(eye).getAttribute('src')) shown = null; };
});
function render(s) {
  $('active').textContent = s.active || 'none';
  $('state').textContent = s.state;
  $('error').textContent = s.error || '';
  var names = Object.keys(s.cameras).concat([null]);
  $('buttons').innerHTML = '';
  names.forEach(function (name) {
    var b = document.createElement('button'), c = s.cameras[name];
    b.textContent = name ? name + ' (' + c.model + ', ' + c.role + ')' : 'release';
    b.className = name === s.active ? 'on' : '';
    b.disabled = busy;
    b.onclick = function () { select(name); };
    $('buttons').appendChild(b);
  });
  $('stats').textContent = JSON.stringify(s, null, 1);
  show(s);
}
function refresh() { fetch('/status').then(function (r) { return r.json(); }).then(render).catch(function () {}); }
function select(name) {
  busy = true; $('msg').textContent = 'switching to ' + (name || 'none') + '...'; refresh();
  fetch('/select', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({camera: name})})
    .then(function (r) { return r.json(); })
    .then(function (j) { $('msg').textContent = j.ok ? '' : 'select failed: ' + j.error; })
    .catch(function (e) { $('msg').textContent = 'select failed: ' + e; })
    .then(function () { busy = false; refresh(); });
}
refresh(); setInterval(refresh, 2000);
</script></body></html>"""


class HubServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 64           # listen() backlog; the default 5 drops connections in a burst of clients

    def __init__(self, address, hub):
        self.hub = hub
        super().__init__(address, HubHandler)


class HubHandler(BaseHTTPRequestHandler):
    server_version = 'camera_hub/%d' % HUB_VERSION
    protocol_version = 'HTTP/1.0'     # a feed ends by closing the connection, like the old server's
    timeout = 20                      # a client that stops reading loses its feed instead of pinning a thread

    # -- plumbing ---------------------------------------------------------------------------------
    def log_request(self, code='-', size='-'):
        if not isinstance(code, int) or code >= 400:
            super().log_request(code, size)

    def log_message(self, fmt, *args):
        log.info('%s %s', self.address_string(), fmt % args)

    def _send(self, code, body, content_type, headers=None):
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj).encode(), 'application/json')

    def _text(self, code, text):
        self._send(code, text.encode(), 'text/plain; charset=utf-8')

    # -- routing ----------------------------------------------------------------------------------
    def do_GET(self):
        url = urllib.parse.urlsplit(self.path)
        path = url.path
        hub = self.server.hub
        if path == '/':
            return self._send(200, PAGE.encode(), 'text/html; charset=utf-8')
        if path == '/status':
            return self._json(200, hub.status())
        if path == '/time':
            return self._json(200, {'t_ns': time.time_ns(), 'monotonic_ns': time.monotonic_ns()})
        if path.startswith('/cameras/'):
            name, _, rest = path[len('/cameras/'):].partition('/')
            camera = hub.cameras.get(name)
            if camera is None:
                return self._json(404, {'error': 'unknown camera %r' % name, 'cameras': sorted(hub.cameras)})
            return self._camera_route(camera, rest, legacy=False)
        if LEGACY_CAMERA in hub.cameras:
            return self._camera_route(hub.cameras[LEGACY_CAMERA], path[1:], legacy=True)
        return self._text(404, 'not found')

    def _camera_route(self, camera, rest, legacy):
        if rest == 'info':
            if legacy and not camera.active:
                return self._inactive(camera, legacy)
            return self._json(200, self._info(camera))
        if rest == 'calibration.conf':
            text = camera.calibration.text()
            if not text:
                return self._text(404, 'no calibration available for SN%d: not downloadable yet (retrying)' % camera.serial)
            return self._send(200, text.encode(), 'text/plain; charset=utf-8')
        if rest == 'stats':
            return self._json(200, camera.legacy_stats() if legacy else camera.status())
        if rest == 'video_feed' or rest.startswith('video_feed/'):
            eye = rest[len('video_feed/'):] or 'left'
            if eye not in FEED_EYES:
                return self._text(404, 'unknown sensor %r; have %s' % (eye, sorted(FEED_EYES)))
            return self._feed(camera, eye, legacy)
        if rest == 'snapshot.jpg' or (rest.startswith('snapshot/') and rest.endswith('.jpg')):
            eye = rest[len('snapshot/'):-len('.jpg')] if rest.startswith('snapshot/') else 'left'
            if eye not in EYES:
                return self._text(404, 'unknown sensor %r; have %s' % (eye, list(EYES)))
            return self._snapshot(camera, eye, legacy)
        return self._text(404, 'not found')

    def _info(self, camera):
        hub = self.server.hub
        return dict(camera.info(), camera=camera.name, active=camera.active, hub_active=hub.active,
                    calibration_available=camera.calibration.available)

    def _inactive(self, camera, legacy):
        active = self.server.hub.active
        if legacy:
            host = self.headers.get('Host') or '%s:%d' % self.server.server_address[:2]
            error = '%s is not the active camera (active: %s); select it at http://%s/' % (camera.model, active, host)
        else:
            error = 'camera %s is not active' % camera.name
        return self._json(503, {'error': error, 'camera': camera.name, 'active': active})

    def _extra_headers(self, camera, legacy):
        return {} if legacy else {'X-Camera': camera.name, 'X-Serial': str(camera.serial)}

    def _snapshot(self, camera, eye, legacy):
        if not camera.active:
            return self._inactive(camera, legacy)
        jpg, meta = camera.latest(eye, -1, timeout=2.0)
        if jpg is None:
            return self._text(503, 'no frame yet')
        headers = frame_headers(meta, eye)
        headers.update(self._extra_headers(camera, legacy))
        return self._send(200, jpg, 'image/jpeg', headers)

    def _feed(self, camera, eye, legacy):
        activation = camera.activation
        if not camera.is_live(activation):
            return self._inactive(camera, legacy)

        def alive():
            return camera.is_live(activation)
        parts = camera.stereo_pairs(alive) if eye == 'stereo' else camera.frames(eye, alive)
        extra = self._extra_headers(camera, legacy)
        started, sent = time.monotonic(), 0
        log.info('%s opened %s/%s', self.address_string(), camera.name, eye)
        try:
            self.send_response(200)
            self.send_header('Content-Type', MULTIPART)
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            for item in parts:
                if item is None:
                    # Idle keep-alive: pushes the response headers out promptly and lets the server
                    # notice a client that went away while the camera produces nothing.
                    self.wfile.write(b'\r\n')
                elif eye == 'stereo':
                    self.wfile.write(stereo_part(*item, extra=extra))
                    sent += 1
                else:
                    self.wfile.write(jpeg_part(item[0], item[1], eye, extra))
                    sent += 1
            reason = 'camera deactivated'
        except OSError as exc:   # BrokenPipe, ConnectionReset, socket.timeout: the client went away
            reason = type(exc).__name__
        finally:
            parts.close()
        log.info('%s closed %s/%s after %.1f s, %d parts (%s)', self.address_string(), camera.name, eye,
                 time.monotonic() - started, sent, reason)

    def do_POST(self):
        url = urllib.parse.urlsplit(self.path)
        if url.path != '/select':
            return self._text(404, 'not found')
        hub = self.server.hub
        query = urllib.parse.parse_qs(url.query)
        try:
            length = int(self.headers.get('Content-Length') or 0)
            body = json.loads(self.rfile.read(length)) if length else {}
            if not isinstance(body, dict):
                raise ValueError('the body must be a JSON object')
            if 'camera' in body:
                name = body['camera']
            elif 'camera' in query:
                name = query['camera'][0]
            else:
                raise ValueError('missing "camera"')
            if name is not None and not isinstance(name, str):
                raise ValueError('"camera" must be a string or null')
            wait = query.get('wait', ['1'])[0] not in ('0', 'false', 'no')
            timeout = float(query.get('timeout', ['30'])[0])
        except (ValueError, TypeError) as exc:
            return self._json(400, {'ok': False, 'error': 'bad request: %s' % exc})
        if name in ('none', ''):
            name = None
        if name is not None and name not in hub.cameras:
            return self._json(400, {'ok': False, 'error': 'unknown camera %r; choose one of %s or null'
                                    % (name, ', '.join(sorted(hub.cameras)))})
        t0 = time.monotonic()
        result = hub.select(name, wait=wait, timeout=timeout)
        status = hub.status()
        elapsed = round(time.monotonic() - t0, 3)
        if result == 'ok':
            return self._json(200, {'ok': True, 'elapsed_s': elapsed, 'status': status})
        errors = {'error': (502, status['error']), 'timeout': (504, 'no frames within %.0f s' % timeout),
                  'superseded': (409, 'selection changed to %s while waiting' % status['active']),
                  'closed': (503, 'the hub is shutting down')}
        code, error = errors[result]
        return self._json(code, {'ok': False, 'error': error, 'elapsed_s': elapsed, 'status': status})


def build_cameras(args):
    zedx_calibration = args.zedx_calibration or '/usr/local/zed/settings/SN%d.conf' % args.zedx_serial
    zedx = ZedXCamera(args.zedx_serial, Calibration(args.zedx_serial, args.cache_dir, 'SVGA', zedx_calibration),
                      stream_fps=args.zedx_stream_fps, quality=args.quality, cv_threads=max(0, args.cv_threads))
    devices = [d.strip() for d in args.nano_devices.split(',') if d.strip()]
    section = {(1920, 1200): 'FHD1200', (1920, 1080): 'FHD', (960, 600): 'SVGA'}.get((args.nano_width, args.nano_height))
    if section is None:
        raise SystemExit('--nano-width/--nano-height %dx%d has no calibration section' % (args.nano_width, args.nano_height))
    nano = NanoCamera(devices, args.nano_serial, Calibration(args.nano_serial, args.cache_dir, section, args.nano_calibration),
                      width=args.nano_width, height=args.nano_height, fps=args.nano_fps, out_width=args.nano_out_width,
                      quality=args.quality, bayer=args.nano_bayer, ae=not args.nano_no_ae, ae_target=args.nano_ae_target,
                      wb=not args.nano_no_wb, exposure=args.nano_exposure, gain=args.nano_gain)
    return [zedx, nano]


def build_server(hub, host='0.0.0.0', port=8090):
    return HubServer((host, port), hub)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8090)
    parser.add_argument('--default-camera', default='zedx_nano', choices=('zedx', 'zedx_nano', 'none'),
                        help='camera selected on the first start; later starts restore the last selection')
    parser.add_argument('--state-file', default=os.path.join(HUB_DIR, 'state.json'), help='persisted selection')
    parser.add_argument('--cache-dir', default=HUB_DIR, help='where downloaded calibration files are cached')
    parser.add_argument('--quality', type=int, default=80, help='JPEG quality')
    parser.add_argument('--cv-threads', type=int, default=1,
                        help='OpenCV worker threads; 1 uses ~4x less CPU than the default pool on Xavier')
    zedx = parser.add_argument_group('ZED X (ZED SDK)')
    zedx.add_argument('--zedx-serial', type=int, default=42757821)
    zedx.add_argument('--zedx-stream-fps', type=float, default=30.0,
                      help='pairs per second to encode and serve (the camera runs at 60 fps)')
    zedx.add_argument('--zedx-calibration', help='.conf file (default /usr/local/zed/settings/SN<serial>.conf, '
                                                 'else downloaded by serial)')
    nano = parser.add_argument_group('ZED X Nano (raw V4L2)')
    nano.add_argument('--nano-devices', default='/dev/video3,/dev/video2',
                      help='left,right V4L2 nodes (the driver\'s video2 is the right sensor)')
    nano.add_argument('--nano-serial', type=int, default=99292912)
    nano.add_argument('--nano-calibration', help='.conf file (default SN<serial>.conf in --cache-dir, else downloaded)')
    nano.add_argument('--nano-width', type=int, default=1920)
    nano.add_argument('--nano-height', type=int, default=1200)
    nano.add_argument('--nano-fps', type=int, default=30, choices=(15, 30, 60))
    nano.add_argument('--nano-out-width', type=int, default=960, help='JPEG width (0 = native)')
    nano.add_argument('--nano-bayer', default='GB', choices=sorted(BAYER_CODES), help='OpenCV Bayer pattern code; GB matches BA10/GRBG')
    nano.add_argument('--nano-no-ae', action='store_true', help='disable software auto exposure')
    nano.add_argument('--nano-ae-target', type=float, default=105.0, help='target mean brightness 0-255')
    nano.add_argument('--nano-no-wb', action='store_true', help='disable gray-world white balance')
    nano.add_argument('--nano-exposure', type=int, default=8000, help='initial exposure in microseconds')
    nano.add_argument('--nano-gain', type=int, default=200, help='initial gain, 100 = 1x')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', stream=sys.stderr)
    cv2.setNumThreads(max(0, args.cv_threads))
    cameras = build_cameras(args)
    hub = CameraHub(cameras, args.state_file, default=None if args.default_camera == 'none' else args.default_camera)
    server = build_server(hub, args.host, args.port)
    # A plain flag: setting an Event from a signal handler can deadlock on the Event's own lock.
    signals = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, frame: signals.append(signum))
    # Load the calibration files in the background: a download (missing file) may take up to 15 s.
    for camera in cameras:
        threading.Thread(target=camera.calibration.text, daemon=True).start()
    hub.start()
    threading.Thread(target=server.serve_forever, name='http', daemon=True).start()
    log.info('serving %s on http://%s:%d/ (active: %s)', ', '.join(c.name for c in cameras), args.host, args.port, hub.active)
    while not signals:
        time.sleep(0.2)
    log.info('stopping')
    server.shutdown()            # no new requests; the open feeds end with their camera below
    hub.close()                  # ends the feeds and releases the camera (zedx_capture.py closes it, v4l2-ctl killed)
    server.server_close()


if __name__ == '__main__':
    main()
