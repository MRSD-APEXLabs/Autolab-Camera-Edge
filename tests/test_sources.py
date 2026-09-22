import io
import threading
import time

import sources
from hub_fakes import CONF
from sources import Calibration, CameraSource, FrameSlot


def test_frame_slot_waits_for_new_frames_and_reset_wakes_waiters():
    slot = FrameSlot()
    assert slot.latest(0, timeout=0.05) == (None, slot.meta)
    assert slot.publish(b'a', 10, 20) == 1
    payload, meta = slot.latest(0, timeout=0)
    assert payload == b'a' and meta['frame_id'] == 1 and meta['capture_ns'] == 10 and meta['capture_mono_ns'] == 20
    assert meta['ready_ns'] > 0 and meta['ready_mono_ns'] > 0
    threading.Timer(0.05, slot.publish, args=(b'b', 11, 21)).start()
    payload, meta = slot.latest(1, timeout=2.0)
    assert payload == b'b' and meta['frame_id'] == 2
    threading.Timer(0.05, slot.reset).start()
    t0 = time.monotonic()
    assert slot.latest(2, timeout=2.0)[0] is None and time.monotonic() - t0 < 1.0
    assert slot.publish(b'c', 12, 22) == 3    # ids keep counting after a reset


def test_activation_drops_a_frame_left_over_from_the_last_session():
    slot = FrameSlot()
    camera = CameraSource('cam', 'Model', 1, 'base', None, [slot])
    camera.activate()
    slot.publish(b'a', 1, 1)
    camera.deactivate()
    assert slot.payload is None
    slot.publish(b'late', 2, 2)       # the capture was still being stopped
    camera.activate()
    assert slot.latest(0, timeout=0)[0] is None and slot.frame_id == 2


def test_calibration_prefers_the_file_then_the_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(sources.urllib.request, 'urlopen', lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    path = tmp_path / 'factory.conf'
    path.write_text(CONF)
    assert Calibration(1, str(tmp_path), 'SVGA', str(path)).text() == CONF
    (tmp_path / 'SN7.conf').write_text(CONF + '# cached\n')
    assert Calibration(7, str(tmp_path), 'SVGA', str(tmp_path / 'missing.conf')).text().endswith('# cached\n')
    assert Calibration(7, str(tmp_path), 'FHD1200').available


def test_calibration_downloads_caches_and_retries_lazily(tmp_path, monkeypatch):
    replies = [OSError('offline'), '[STEREO]\nBaseline=120\n', CONF]
    urls = []

    def urlopen(url, timeout):
        urls.append(url)
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return io.BytesIO(reply.encode())
    monkeypatch.setattr(sources.urllib.request, 'urlopen', urlopen)
    calibration = Calibration(42757821, str(tmp_path), 'SVGA', retry_interval=0.05)
    assert calibration.text() is None and not calibration.available   # second call within the retry interval
    assert urls == ['https://calib.stereolabs.com/?SN=42757821']
    time.sleep(0.06)
    assert calibration.text() is None       # a file without the streamed mode's section is useless
    time.sleep(0.06)
    assert calibration.text() == CONF and (tmp_path / 'SN42757821.conf').read_text() == CONF
    assert len(urls) == 3
    assert Calibration(0, str(tmp_path), 'SVGA').text() is None
