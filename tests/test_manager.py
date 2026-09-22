import json
import threading
import time

import pytest

from hub_fakes import FakeNano, FakeZedX, wait_for
from manager import ERROR, IDLE, STARTING, STREAMING, CameraHub


@pytest.fixture
def make_hub(tmp_path):
    hubs = []

    def make(default='zedx_nano', **kwargs):
        zedx, nano = FakeZedX(tmp_path), FakeNano(tmp_path)
        options = dict(stall_s=0.5, start_timeout_s=2.0, retry_delays_s=(0.1, 0.2), stop_timeout_s=2.0, tick_s=0.02)
        options.update(kwargs)
        hub = CameraHub([zedx, nano], str(tmp_path / 'state.json'), default=default, **options)
        hubs.append(hub)
        return hub, zedx, nano
    yield make
    for hub in hubs:
        hub.close()


def saved(hub):
    with open(hub.state_path) as f:
        return json.load(f)


def test_first_start_uses_the_default_and_later_starts_the_saved_selection(make_hub, tmp_path):
    hub, zedx, nano = make_hub()
    hub.start()
    assert hub.active == 'zedx_nano' and nano.active and not zedx.active
    assert wait_for(lambda: hub.state == STREAMING)
    assert not (tmp_path / 'state.json').exists()   # nothing was selected yet
    assert hub.select('zedx') == 'ok'
    hub.close()
    assert saved(hub) == {'active': 'zedx'}
    hub, zedx, nano = make_hub()
    hub.start()
    assert hub.active == 'zedx' and zedx.active and not nano.active
    assert hub.select(None) == 'ok'
    hub.close()
    hub, _, _ = make_hub()
    hub.start()
    assert hub.active is None and hub.state == IDLE


@pytest.mark.parametrize('content', ['not json', '{"active": "zedx_mini"}', '[]'])
def test_unusable_state_file_falls_back_to_the_default(make_hub, tmp_path, content):
    (tmp_path / 'state.json').write_text(content)
    hub, _, _ = make_hub()
    hub.start()
    assert hub.active == 'zedx_nano'


def test_select_switches_releases_and_persists(make_hub):
    hub, zedx, nano = make_hub()
    hub.start()
    assert wait_for(lambda: hub.state == STREAMING)
    nano_session = nano.sessions[0]
    assert hub.select('zedx') == 'ok'
    assert hub.active == 'zedx' and hub.state == STREAMING and hub.switch_count == 1
    assert zedx.active and not nano.active
    assert nano_session.stopped_at is not None and not nano_session.is_alive()
    # nothing published once it was stopped (a frame from during the stop is dropped by the next activate())
    assert nano.eye_slots['left'].updated <= nano_session.stopped_at
    assert saved(hub) == {'active': 'zedx'}
    status = hub.status()
    assert status['active'] == 'zedx' and status['state'] == 'streaming' and status['error'] is None
    assert status['cameras']['zedx']['active'] and not status['cameras']['zedx_nano']['active']
    assert status['cameras']['zedx']['frames'] > 0 and status['cameras']['zedx']['backend'] == 'zed_sdk'
    assert status['cameras']['zedx_nano']['backend'] == 'v4l2' and status['cameras']['zedx_nano']['role'] == 'wrist'
    assert hub.select(None) == 'ok'
    assert hub.active is None and hub.state == IDLE and not zedx.active
    assert not zedx.sessions[-1].is_alive() and saved(hub) == {'active': None} and hub.switch_count == 2


def test_selecting_the_healthy_active_camera_is_a_no_op(make_hub):
    hub, _, nano = make_hub()
    hub.start()
    assert hub.select('zedx_nano') == 'ok'
    assert len(nano.sessions) == 1 and hub.switch_count == 0
    with pytest.raises(KeyError):
        hub.select('zedx_mini')


def test_failed_start_reports_the_error_and_retries_until_it_works(make_hub):
    hub, zedx, _ = make_hub()
    hub.start()
    zedx.fail_starts = 2
    t0 = time.monotonic()
    assert hub.select('zedx') == 'error'
    assert hub.active == 'zedx' and hub.state == ERROR and hub.error == 'fake open failed'
    assert hub.status()['retry_in_s'] is not None
    assert wait_for(lambda: hub.state == STREAMING)
    # backoff 0.1 s, then 0.2 s
    assert len(zedx.sessions) == 3 and time.monotonic() - t0 >= 0.3
    assert zedx.restarts == 2 and zedx.last_error == 'fake open failed' and hub.error is None
    assert hub.status()['cameras']['zedx']['restarts'] == 2


def test_selecting_a_failed_camera_again_retries_at_once(make_hub):
    hub, zedx, _ = make_hub(retry_delays_s=(60.0,))
    hub.start()
    zedx.fail_starts = 1
    assert hub.select('zedx') == 'error'
    assert hub.select('zedx') == 'ok' and len(zedx.sessions) == 2 and hub.switch_count == 1


def test_stall_sets_error_and_recovery_clears_it(make_hub):
    hub, _, nano = make_hub()
    hub.start()
    assert wait_for(lambda: hub.state == STREAMING)
    nano.stall.set()
    assert wait_for(lambda: hub.state == ERROR)
    assert hub.error.startswith('no frames for')
    nano.stall.clear()
    assert wait_for(lambda: hub.state == STREAMING)
    assert len(nano.sessions) == 1    # the Nano's sensors recover on their own; the session is kept


def test_no_first_frame_times_out_but_keeps_the_session(make_hub):
    hub, zedx, _ = make_hub(start_timeout_s=0.3)
    hub.start()
    zedx.start_delay = 1.0
    assert hub.select('zedx', timeout=5) == 'error'
    assert 'no frames' in hub.error
    assert wait_for(lambda: hub.state == STREAMING) and len(zedx.sessions) == 1


def test_select_wait_timeout_and_no_wait(make_hub):
    hub, zedx, nano = make_hub()
    hub.start()
    zedx.start_delay = 1.0
    t0 = time.monotonic()
    assert hub.select('zedx', timeout=0.2) == 'timeout'
    assert time.monotonic() - t0 < 1.0 and hub.state == STARTING
    assert hub.select('zedx_nano', wait=False) == 'ok'
    assert hub.active == 'zedx_nano' and hub.state == STARTING


def test_a_newer_selection_supersedes_a_waiting_one(make_hub):
    hub, zedx, _ = make_hub()
    hub.start()
    zedx.start_delay = 2.0
    results = []
    waiter = threading.Thread(target=lambda: results.append(hub.select('zedx')))
    waiter.start()
    assert wait_for(lambda: hub.active == 'zedx')
    assert hub.select('zedx_nano') == 'ok'
    waiter.join(5)
    assert results == ['superseded'] and hub.switch_count == 2


def test_frame_ids_keep_counting_across_activations(make_hub):
    hub, zedx, _ = make_hub()
    hub.start()
    hub.select('zedx')
    assert wait_for(lambda: zedx.pairs.frame_id >= 3)
    hub.select('zedx_nano')
    first = zedx.pairs.frame_id
    hub.select('zedx')
    _, meta = zedx.latest('left', -1, timeout=2.0)
    assert meta['frame_id'] > first


def test_feeds_end_when_their_camera_is_deactivated(make_hub):
    hub, zedx, nano = make_hub()
    hub.start()
    hub.select('zedx')
    activation = zedx.activation
    pairs = zedx.stereo_pairs(lambda: zedx.is_live(activation))
    left = zedx.frames('left', lambda: zedx.is_live(activation))
    assert next(pairs) is not None and next(left) is not None
    ended = {}

    def drain(name, gen):
        for _ in gen:
            pass
        ended[name] = time.monotonic()
    threads = [threading.Thread(target=drain, args=item) for item in (('pairs', pairs), ('left', left))]
    for thread in threads:
        thread.start()
    time.sleep(0.2)
    t0 = time.monotonic()
    hub.select('zedx_nano', wait=False)
    for thread in threads:
        thread.join(3)
    assert set(ended) == {'pairs', 'left'} and max(ended.values()) - t0 < 1.0
    # a later reactivation doesn't revive an old feed
    assert not zedx.is_live(activation)


def test_nano_pairs_match_capture_stamps(make_hub):
    hub, _, nano = make_hub()
    hub.start()
    pairs = nano.stereo_pairs(lambda: True)
    for _ in range(3):
        item = next(pairs)
        while item is None:
            item = next(pairs)
        ljpg, lmeta, rjpg, rmeta = item
        assert lmeta['capture_mono_ns'] - rmeta['capture_mono_ns'] == -1_000_000
        assert ljpg != rjpg


def test_a_selection_during_close_opens_nothing(make_hub, tmp_path):
    hub, zedx, nano = make_hub()
    hub.start()
    assert wait_for(lambda: hub.state == STREAMING)
    session = nano.sessions[0]
    stop = session.stop
    session.stop = lambda timeout=5.0: time.sleep(0.5) or stop(timeout)    # slow like zed.close()
    closer = threading.Thread(target=hub.close)
    closer.start()
    time.sleep(0.1)
    assert hub.select('zedx', wait=False) == 'closed'
    closer.join(5)
    assert zedx.sessions == [] and not zedx.active and hub.active == 'zedx_nano' and hub.state == IDLE
    assert not (tmp_path / 'state.json').exists()


def test_close_ends_a_waiting_selection(make_hub):
    hub, zedx, _ = make_hub()
    hub.start()
    zedx.start_delay = 5.0
    results = []
    waiter = threading.Thread(target=lambda: results.append(hub.select('zedx')))
    waiter.start()
    assert wait_for(lambda: hub.active == 'zedx')
    t0 = time.monotonic()
    hub.close()
    waiter.join(5)
    assert results == ['closed'] and time.monotonic() - t0 < 2.0
