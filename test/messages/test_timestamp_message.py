from decision_making.src.messages.scene_common_messages import Timestamp


def test_timestampMessage_toFromSeconds_accurate():
    timestamps_in_secs = [9245.567123, 0.0, 3.0, 1.5]
    timestamp_messages = [Timestamp.from_seconds(timestamp_in_sec) for timestamp_in_sec in timestamps_in_secs]
    timestamps_parsed = [timestamp_message.timestamp_in_seconds() for timestamp_message in timestamp_messages]
    assert timestamps_parsed == timestamps_in_secs


