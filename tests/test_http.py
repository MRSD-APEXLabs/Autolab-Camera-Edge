import signal
import socket
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import pytest

import camera_hub
from hub_fakes import CONF, Client, FakeNano, FakeZedX, read_parts, wait_for
from manager import CameraHub

NANO_INFO_KEYS = {'server_version', 'model', 'serial', 'capture', 'output', 'sensors', 'stereo', 'processing',
                  'calibration_available'}


@pytest.fixture
def hub_server(tmp_path):
    """A hub on an ephemeral port with fake cameras; the Nano calibration is cached, the ZED X one is a file."""
    (tmp_path / 'SN99292912.conf').write_text(CONF)
    zedx_conf = tmp_path / 'zedx.conf'
    zedx_conf.write_text(CONF)
    zedx, nano = FakeZedX(tmp_path, str(zedx_conf)), FakeNano(tmp_path)
    hub = CameraHub([zedx, nano], str(tmp_path / 'state.json'), default='zedx_nano', stall_s=0.5,
                    retry_delays_s=(0.2,), stop_timeout_s=2.0, tick_s=0.02)
    server = camera_hub.build_server(hub, '127.0.0.1', 0)
    hub.start()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = Client(server.server_address[1])
    assert wait_for(lambda: hub.state == 'streaming')
    yield client, hub, zedx, nano
    hub.close()
    server.shutdown()
    server.server_close()


def decode(jpg):
    image = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    return image


def test_status_time_and_page(hub_server):
    client, hub, _, _ = hub_server
    code, status = client.json('GET', '/status')
    assert code == 200 and status['server'] == 'camera_hub' and status['hub_version'] == 1
    assert status['active'] == 'zedx_nano' and status['state'] == 'streaming' and status['error'] is None
    assert status['switch_count'] == 0 and status['state_since_s'] >= 0
    assert set(status['cameras']) == {'zedx', 'zedx_nano'}
    zedx = status['cameras']['zedx']
    assert (zedx['model'], zedx['serial'], zedx['role'], zedx['backend'], zedx['active']) == \
        ('ZED X', 42757821, 'base', 'zed_sdk', False)
    for key in ('fps', 'frames', 'restarts', 'last_error'):
        assert key in zedx and key in status['cameras']['zedx_nano']
    assert abs(status['t_ns'] - time.time_ns()) < 5e9
    code, clock = client.json('GET', '/time')
    assert code == 200 and set(clock) == {'t_ns', 'monotonic_ns'}
    code, content_type, body, _ = client.get('/')
    assert code == 200 and content_type.startswith('text/html') and b'/select' in body


def test_select_over_http(hub_server):
    client, hub, zedx, _ = hub_server
    code, reply = client.select('zedx')
    assert code == 200 and reply['ok'] and reply['status']['active'] == 'zedx'
    assert reply['status']['state'] == 'streaming' and reply['status']['switch_count'] == 1
    code, reply = client.select('zedx')                 # already active and healthy
    assert code == 200 and reply['status']['switch_count'] == 1
    code, reply = client.select('zedx_mini')
    assert code == 400 and not reply['ok'] and 'zedx_mini' in reply['error']
    code, reply = client.json('POST', '/select', {'nothing': 1})
    assert code == 400
    conn, response = client.request('POST', '/select')
    assert response.status == 400
    conn.close()
    code, reply = client.select('none')
    assert code == 200 and reply['status']['active'] is None and reply['status']['state'] == 'idle'
    code, reply = client.json('POST', '/select?camera=zedx_nano&wait=0')   # query form, no body
    assert code == 200 and reply['status']['active'] == 'zedx_nano' and reply['status']['state'] == 'starting'


def test_select_failure_and_timeout(hub_server):
    client, hub, zedx, _ = hub_server
    zedx.fail_starts = 1
    code, reply = client.select('zedx')
    assert code == 502 and not reply['ok'] and reply['error'] == 'fake open failed'
    assert reply['status']['active'] == 'zedx' and reply['status']['state'] == 'error'
    assert wait_for(lambda: hub.state == 'streaming')    # retried in the background
    client.select('zedx_nano')
    zedx.start_delay = 3.0
    code, reply = client.select('zedx', '?timeout=0.3')
    assert code == 504 and not reply['ok'] and reply['status']['state'] == 'starting'
    code, _ = client.json('POST', '/select?timeout=soon', {'camera': 'zedx'})
    assert code == 400


@pytest.mark.parametrize('camera', ['zedx', 'zedx_nano'])
def test_stereo_feed(hub_server, camera):
    client, hub, _, _ = hub_server
    client.select(camera)
    conn, response = client.request('GET', '/cameras/%s/video_feed/stereo' % camera)
    assert response.status == 200 and response.getheader('Content-Type') == 'multipart/x-mixed-replace; boundary=frame'
    parts = read_parts(response)
    ids = []
    for _ in range(4):
        headers, body = next(parts)
        assert headers['content-type'] == 'application/octet-stream' and headers['x-sensor'] == 'stereo'
        assert headers['x-camera'] == camera and headers['x-serial'] == str(hub.cameras[camera].serial)
        left_length = int(headers['x-left-length'])
        left, right = decode(body[:left_length]), decode(body[left_length:])
        assert left.shape == right.shape == (6, 8, 3) and left.mean() < right.mean()
        for key in ('x-capture-ns', 'x-ready-ns', 'x-capture-mono-ns', 'x-ready-mono-ns', 'x-right-capture-ns'):
            assert int(headers[key]) > 0
        if camera == 'zedx':   # one grab: shared id and stamp
            assert headers['x-right-frame-id'] == headers['x-frame-id'] and headers['x-sync-us'] == '0'
            assert headers['x-right-capture-ns'] == headers['x-capture-ns']
        else:
            assert headers['x-sync-us'] == '-1000'
        ids.append(int(headers['x-frame-id']))
    assert ids == sorted(set(ids))
    conn.close()


def test_a_superseded_selection_answers_409(hub_server):
    client, hub, zedx, _ = hub_server
    zedx.start_delay = 2.0
    replies = []
    waiter = threading.Thread(target=lambda: replies.append(client.select('zedx')))
    waiter.start()
    assert wait_for(lambda: hub.active == 'zedx')
    code, reply = client.select('zedx_nano')
    waiter.join(5)
    assert code == 200 and reply['ok']
    assert replies[0][0] == 409 and replies[0][1]['error'] == 'selection changed to zedx_nano while waiting'


def test_select_after_close_answers_503(hub_server):
    client, hub, _, _ = hub_server
    hub.close()
    code, reply = client.select('zedx')
    assert code == 503 and reply['ok'] is False and reply['error'] == 'the hub is shutting down'
    assert reply['status']['active'] == 'zedx_nano' and reply['status']['state'] == 'idle'


def test_status_tells_the_page_when_its_feeds_ended(hub_server):
    """A release and reselect between two page polls ends the feeds without changing active/state."""
    client, hub, _, _ = hub_server
    _, status = client.json('GET', '/status')
    started, activation = status['started_ns'], status['cameras']['zedx_nano']['activation']
    client.select(None)
    client.select('zedx_nano')
    _, status = client.json('GET', '/status')
    assert (status['active'], status['state']) == ('zedx_nano', 'streaming')
    assert status['started_ns'] == started and status['cameras']['zedx_nano']['activation'] == activation + 1
    page = client.get('/')[2].decode()
    assert '[s.active, c.activation, s.started_ns]' in page and '.onerror' in page


def test_idle_feed_sends_keep_alives(hub_server):
    client, hub, _, nano = hub_server
    nano.stall.set()
    time.sleep(0.1)
    conn, response = client.request('GET', '/cameras/zedx_nano/video_feed/left')
    headers, _ = next(read_parts(response))          # the last frame before the stall
    t0 = time.monotonic()
    lines = [response.readline() for _ in range(3)]  # the part's closing CRLF, then one CRLF per idle second
    elapsed = time.monotonic() - t0
    conn.close()
    assert lines == [b'\r\n'] * 3 and 1.5 < elapsed < 3.5


def test_single_eye_feeds_and_snapshots(hub_server):
    client, hub, _, _ = hub_server
    for camera in ('zedx_nano', 'zedx'):
        client.select(camera)
        for eye, brighter in (('left', False), ('right', True)):
            conn, response = client.request('GET', '/cameras/%s/video_feed/%s' % (camera, eye))
            headers, body = next(read_parts(response))
            conn.close()
            assert headers['content-type'] == 'image/jpeg' and headers['x-sensor'] == eye
            assert headers['x-camera'] == camera
            assert (decode(body).mean() > 120) == brighter
            code, content_type, body, headers = client.get('/cameras/%s/snapshot/%s.jpg' % (camera, eye))
            assert code == 200 and content_type == 'image/jpeg' and headers['X-Sensor'] == eye
            assert int(headers['X-Frame-Id']) > 0 and headers['X-Camera'] == camera
            assert (decode(body).mean() > 120) == brighter
        code, stats = client.json('GET', '/cameras/%s/stats' % camera)
        assert code == 200 and stats['active'] and stats['frames'] > 0
    assert client.get('/cameras/zedx/video_feed/middle')[0] == 404
    assert client.get('/cameras/zedx/snapshot/stereo.jpg')[0] == 404
    code, reply = client.json('GET', '/cameras/zedx_mini/info')
    assert code == 404 and reply['cameras'] == ['zedx', 'zedx_nano']


def test_inactive_camera_answers_503(hub_server):
    client, hub, _, _ = hub_server
    for path in ('/cameras/zedx/video_feed/stereo', '/cameras/zedx/video_feed/left', '/cameras/zedx/snapshot/right.jpg'):
        code, reply = client.json('GET', path)
        assert code == 503 and reply == {'error': 'camera zedx is not active', 'camera': 'zedx', 'active': 'zedx_nano'}
    code, info = client.json('GET', '/cameras/zedx/info')     # info works either way
    assert code == 200 and info['active'] is False and info['hub_active'] == 'zedx_nano' and info['camera'] == 'zedx'
    assert info['capture'] == {'width': 960, 'height': 600, 'fps': 60, 'pixel_format': 'zed_sdk'}
    assert info['output'] == {'width': 960, 'height': 600, 'format': 'jpeg'}
    assert info['processing'] == {'isp': True, 'rectified': False, 'stream_fps': 30}
    assert info['stereo']['available'] and info['server_version'] == 4 and info['calibration_available']
    code, _, body, _ = client.get('/cameras/zedx/calibration.conf')
    assert code == 200 and body.decode() == CONF


def test_open_feed_ends_within_a_second_of_a_switch(hub_server):
    client, hub, _, _ = hub_server
    client.select('zedx')
    conn, response = client.request('GET', '/cameras/zedx/video_feed/stereo')
    parts = read_parts(response)
    next(parts)
    ended = []

    def drain():
        for _ in parts:
            pass
        ended.append(time.monotonic())
    thread = threading.Thread(target=drain)
    thread.start()
    t0 = time.monotonic()
    client.select('zedx_nano', '?wait=0')
    thread.join(5)
    conn.close()
    assert ended and ended[0] - t0 < 1.0


def test_legacy_routes_follow_the_nano(hub_server):
    client, hub, _, nano = hub_server
    code, info = client.json('GET', '/info')
    assert code == 200 and NANO_INFO_KEYS <= set(info) and info['active'] is True
    assert info['model'] == 'ZED X Nano' and info['serial'] == 99292912 and info['server_version'] == 4
    assert info['capture'] == {'width': 1920, 'height': 1200, 'fps': 30, 'pixel_format': 'BA10'}
    assert info['sensors'] == {'left': '/dev/fake3', 'right': '/dev/fake2'}
    assert info['stereo']['available'] and info['stereo']['tolerance_us'] == 12000
    conn, response = client.request('GET', '/video_feed/stereo')
    headers, body = next(read_parts(response))
    conn.close()
    assert headers['x-sensor'] == 'stereo' and 'x-camera' not in headers   # byte-compatible with the old server
    conn, response = client.request('GET', '/video_feed')
    headers, _ = next(read_parts(response))
    conn.close()
    assert headers['x-sensor'] == 'left'
    for path in ('/snapshot.jpg', '/snapshot/right.jpg'):
        code, content_type, _, _ = client.get(path)
        assert code == 200 and content_type == 'image/jpeg'
    code, stats = client.json('GET', '/stats')
    assert code == 200 and set(stats) == {'left', 'right'} and stats['left']['device'] == '/dev/fake3'

    client.select('zedx')
    for path in ('/info', '/video_feed', '/video_feed/stereo', '/video_feed/right', '/snapshot.jpg', '/snapshot/left.jpg'):
        code, reply = client.json('GET', path)
        assert code == 503 and reply['camera'] == 'zedx_nano' and reply['active'] == 'zedx'
        assert reply['error'].startswith('ZED X Nano is not the active camera (active: zedx); select it at http://127.0.0.1:')
    code, _, body, _ = client.get('/calibration.conf')
    assert code == 200 and body.decode() == CONF
    assert client.get('/stats')[0] == 200
    assert client.get('/video_feed/middle')[0] == 404
    assert client.get('/favicon.ico')[0] == 404


def test_missing_calibration_is_404(hub_server, monkeypatch):
    client, hub, zedx, _ = hub_server
    zedx.calibration.path = '/nonexistent/SN42757821.conf'
    zedx.calibration._text, zedx.calibration._next_try = None, 0.0
    monkeypatch.setattr('sources.urllib.request.urlopen', lambda *a, **k: (_ for _ in ()).throw(OSError('offline')))
    code, _, body, _ = client.get('/cameras/zedx/calibration.conf')
    assert code == 404 and b'SN42757821' in body
    code, info = client.json('GET', '/cameras/zedx/info')
    assert code == 200 and info['calibration_available'] is False


def test_command_line_defaults():
    args = camera_hub.parse_args([])
    assert (args.host, args.port, args.default_camera, args.cv_threads) == ('0.0.0.0', 8090, 'zedx_nano', 1)
    assert args.nano_devices == '/dev/video3,/dev/video2' and args.zedx_serial == 42757821
    assert args.zedx_stream_fps == 30 and args.quality == 80 and args.state_file.endswith('state.json')
    zedx, nano = camera_hub.build_cameras(args)
    assert zedx.name == 'zedx' and nano.name == 'zedx_nano'
    assert zedx.calibration.path == '/usr/local/zed/settings/SN42757821.conf' and zedx.calibration.section == 'SVGA'
    assert nano.calibration.section == 'FHD1200' and nano.output == [960, 600]
    assert nano.devices == {'left': '/dev/video3', 'right': '/dev/video2'}


def test_main_serves_until_sigterm(tmp_path):
    conf = tmp_path / 'calibration.conf'
    conf.write_text(CONF)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    proc = subprocess.Popen([sys.executable, camera_hub.__file__, '--host', '127.0.0.1', '--port', str(port),
                             '--default-camera', 'none', '--state-file', str(tmp_path / 'state.json'),
                             '--cache-dir', str(tmp_path), '--zedx-calibration', str(conf),
                             '--nano-calibration', str(conf)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    client = Client(port, timeout=2.0)

    def status():
        try:
            return client.json('GET', '/status')
        except OSError:
            return None
    try:
        assert wait_for(status, 20)
        assert status()[1]['active'] is None
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(10) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
    log = proc.stdout.read().decode()
    assert 'stopping' in log and 'Traceback' not in log
