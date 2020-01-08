import numpy as np
from rte.python.logger.AV_logger import AV_Logger
from decision_making.src.messages.pedal_position_message import PedalPosition, DataPedalPosition
from decision_making.src.messages.scene_common_messages import Header, Timestamp
from decision_making.src.messages.scene_static_message import TrafficControlBar
from decision_making.src.planning.utils.generalized_frenet_serret_frame import FrenetSubSegment, \
    GeneralizedFrenetSerretFrame
from decision_making.src.scene.scene_static_model import SceneStaticModel
from decision_making.src.utils.map_utils import MapUtils
from decision_making.test.messages.scene_static_fixture import scene_static_pg_split
from decision_making.src.planning.behavioral.state.driver_initiated_motion_state import DriverInitiatedMotionState, \
    DIM_States


def test_update_pedalPressedAndReleased_becomesActiveAndInactive(scene_static_pg_split):
    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)
    for lane_segment in scene_static_pg_split.s_Data.s_SceneStaticBase.as_scene_lane_segments:
        lane_segment.as_traffic_control_bar = []
    scene_static_pg_split.s_Data.s_SceneStaticBase.as_scene_lane_segments[0].as_traffic_control_bar = []
    scene_static_pg_split.s_Data.s_SceneStaticBase.as_static_traffic_control_device = []
    scene_static_pg_split.s_Data.s_SceneStaticBase.as_dynamic_traffic_control_device = []

    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)

    logger = AV_Logger.get_logger()

    # add stop sign to scene_static
    SELECTED_STOP_LANE_ID = 200
    patched_lane = MapUtils.get_lane(lane_id=SELECTED_STOP_LANE_ID)
    stop_bar = TrafficControlBar(e_i_traffic_control_bar_id=1, e_l_station=10., e_i_static_traffic_control_device_id=[],
                                 e_i_dynamic_traffic_control_device_id=[])
    patched_lane.as_traffic_control_bar.append(stop_bar)

    road_id = 20
    lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_id)[0]
    frenet_frame = MapUtils.get_lane_frenet_frame(lane_id)
    sub_segment = FrenetSubSegment(lane_id, 0, frenet_frame.s_max)
    reference_route = GeneralizedFrenetSerretFrame.build([frenet_frame], [sub_segment])

    dim_state = DriverInitiatedMotionState(logger)

    stop_bar_s = 20
    # too far from the stop sign
    lane_fstate = np.array([stop_bar.e_l_station - 7., 0, 0, 0, 0, 0])
    ego_s = stop_bar_s - 7

    # pedal pressed
    pedal_position = create_pedal_position(time_in_sec=0.0, pedal_pos=0.1)
    dim_state.update_pedal_times(pedal_position)
    dim_state.update_state(timestamp_in_sec=pedal_position.s_Data.s_RecvTimestamp.timestamp_in_seconds, ego_lane_fstate=lane_fstate,
                           ego_s=ego_s, closestTCB=(stop_bar, stop_bar_s), ignored_TCB_distance=stop_bar_s)
    assert dim_state.stop_bar_to_ignore() is None and dim_state.state == DIM_States.DISABLED

    # pedal pressed for enough time
    pedal_position = create_pedal_position(time_in_sec=0.2, pedal_pos=0.1)
    dim_state.update_pedal_times(pedal_position)
    dim_state.update_state(timestamp_in_sec=pedal_position.s_Data.s_RecvTimestamp.timestamp_in_seconds, ego_lane_fstate=lane_fstate,
                           ego_s=ego_s, closestTCB=(stop_bar, stop_bar_s), ignored_TCB_distance=stop_bar_s)
    assert dim_state.stop_bar_to_ignore() is None and dim_state.state == DIM_States.DISABLED

    # near the stop sign
    lane_fstate = np.array([stop_bar.e_l_station - 2., 0, 0, 0, 0, 0])
    ego_s = stop_bar_s - 2

    # pedal pressed
    pedal_position = create_pedal_position(time_in_sec=0.4, pedal_pos=0.1)
    dim_state.update_pedal_times(pedal_position)
    dim_state.update_state(timestamp_in_sec=pedal_position.s_Data.s_RecvTimestamp.timestamp_in_seconds, ego_lane_fstate=lane_fstate,
                           ego_s=ego_s, closestTCB=(stop_bar, stop_bar_s), ignored_TCB_distance=stop_bar_s)
    assert dim_state.stop_bar_to_ignore() is None and dim_state.state == DIM_States.PENDING

    # pedal pressed for enough time
    pedal_position = create_pedal_position(time_in_sec=0.6, pedal_pos=0.1)
    dim_state.update_pedal_times(pedal_position)
    dim_state.update_state(timestamp_in_sec=pedal_position.s_Data.s_RecvTimestamp.timestamp_in_seconds, ego_lane_fstate=lane_fstate,
                           ego_s=ego_s, closestTCB=(stop_bar, stop_bar_s), ignored_TCB_distance=stop_bar_s)
    assert dim_state.stop_bar_to_ignore() is not None and dim_state.state == DIM_States.CONFIRMED

    # pedal released
    pedal_position = create_pedal_position(time_in_sec=0.7, pedal_pos=0.01)
    dim_state.update_pedal_times(pedal_position)
    dim_state.update_state(timestamp_in_sec=pedal_position.s_Data.s_RecvTimestamp.timestamp_in_seconds, ego_lane_fstate=lane_fstate,
                           ego_s=ego_s, closestTCB=(stop_bar, stop_bar_s), ignored_TCB_distance=stop_bar_s)
    assert dim_state.stop_bar_to_ignore() is not None and dim_state.state == DIM_States.CONFIRMED

    # crossed the stop sign
    lane_fstate = np.array([stop_bar.e_l_station + 15., 0, 0, 0, 0, 0])
    ego_s = stop_bar_s + 15

    pedal_position = create_pedal_position(time_in_sec=0.8, pedal_pos=0.01)
    dim_state.update_pedal_times(pedal_position)
    dim_state.update_state(timestamp_in_sec=pedal_position.s_Data.s_RecvTimestamp.timestamp_in_seconds, ego_lane_fstate=lane_fstate,
                           ego_s=ego_s, closestTCB=(stop_bar, stop_bar_s), ignored_TCB_distance=stop_bar_s-ego_s)
    assert dim_state.stop_bar_to_ignore() is None and dim_state.state == DIM_States.DISABLED

    # timeout of the DIM state
    pedal_position = create_pedal_position(time_in_sec=3.3, pedal_pos=0.01)
    dim_state.update_pedal_times(pedal_position)
    dim_state.update_state(timestamp_in_sec=pedal_position.s_Data.s_RecvTimestamp.timestamp_in_seconds, ego_lane_fstate=lane_fstate,
                           ego_s=ego_s, closestTCB=(stop_bar, stop_bar_s), ignored_TCB_distance=stop_bar_s-ego_s)
    assert dim_state.stop_bar_to_ignore() is None and dim_state.state == DIM_States.DISABLED


def create_pedal_position(time_in_sec: float, pedal_pos: float):
    return PedalPosition(Header(0, Timestamp.from_seconds(time_in_sec), 0),
                         DataPedalPosition(0, pedal_pos, Timestamp.from_seconds(time_in_sec), True))
