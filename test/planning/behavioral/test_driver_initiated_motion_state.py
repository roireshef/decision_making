from decision_making.src.messages.driver_initiated_motion_message import DriverInitiatedMotionState


def test_update_pedalPressedAndReleased_becomesActiveAndInactive():
    dim_state = DriverInitiatedMotionState()
    dim_state.update(0, pedal_rate=1, stop_bar_lane_id=1, stop_bar_lane_station=10)
    assert not dim_state.is_active()
    # pedal pressed
    dim_state.update(1, pedal_rate=10, stop_bar_lane_id=1, stop_bar_lane_station=20)
    assert not dim_state.is_active()
    # pedal pressed for enough time
    dim_state.update(2, pedal_rate=10, stop_bar_lane_id=1, stop_bar_lane_station=30)
    assert dim_state.is_active()
    # pedal released
    dim_state.update(3, pedal_rate=1, stop_bar_lane_id=1, stop_bar_lane_station=30)
    assert dim_state.is_active()
    assert dim_state.stop_bar_lane_station == 30
    # timeout of the DIM state
    dim_state.update(33, pedal_rate=1, stop_bar_lane_id=1, stop_bar_lane_station=30)
    assert not dim_state.is_active()
