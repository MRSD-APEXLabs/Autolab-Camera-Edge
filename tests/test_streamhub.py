import pytest
from streamhub import StreamHub


def test_wrist_raw_channel_publish_and_snapshot():
    hub = StreamHub()
    hub.publish("wrist_raw", {"image": "abc"})
    seq, payload = hub.snapshot("wrist_raw")
    assert seq == 1
    assert payload["image"] == "abc"
    assert "frame_id" in payload
    assert "timestamp" in payload


def test_base_raw_channel_publish_and_snapshot():
    hub = StreamHub()
    hub.publish("base_raw", {"image": "xyz"})
    seq, payload = hub.snapshot("base_raw")
    assert seq == 1
    assert payload["image"] == "xyz"


def test_unknown_channel_publish_raises():
    hub = StreamHub()
    with pytest.raises(ValueError, match="Unknown channel"):
        hub.publish("nonexistent", {})


def test_unknown_channel_snapshot_raises():
    hub = StreamHub()
    with pytest.raises(ValueError, match="Unknown channel"):
        hub.snapshot("nonexistent")


def test_channels_are_independent():
    hub = StreamHub()
    hub.publish("wrist_raw", {"data": 1})
    hub.publish("base_raw", {"data": 2})
    _, wrist = hub.snapshot("wrist_raw")
    _, base = hub.snapshot("base_raw")
    assert wrist["data"] == 1
    assert base["data"] == 2
