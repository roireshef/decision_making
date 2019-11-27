import numpy as np
from rte.python.logger.AV_logger import AV_Logger
from decision_making.src.messages.pedal_position_message import PedalPosition, DataPedalPosition
from decision_making.src.messages.scene_common_messages import Header, Timestamp
from decision_making.src.messages.scene_static_enums import RoadObjectType
from decision_making.src.messages.scene_static_message import StaticTrafficFlowControl
from decision_making.src.planning.utils.generalized_frenet_serret_frame import FrenetSubSegment, \
    GeneralizedFrenetSerretFrame
from decision_making.src.scene.scene_static_model import SceneStaticModel
from decision_making.src.utils.map_utils import MapUtils
from decision_making.test.messages.scene_static_fixture import scene_static_pg_split
from decision_making.src.planning.behavioral.state.driver_initiated_motion_state import DriverInitiatedMotionState


def test_update_pedalPressedAndReleased_becomesActiveAndInactive(scene_static_pg_split):
    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)

    logger = AV_Logger.get_logger()

    # add stop sign to scene_static
    SELECTED_STOP_LANE_ID = 200
    patched_lane = MapUtils.get_lane(lane_id=SELECTED_STOP_LANE_ID)
    stop_bar = StaticTrafficFlowControl(RoadObjectType.StopSign, e_l_station=10., e_Pct_confidence=100)
    patched_lane.as_static_traffic_flow_control.append(stop_bar)

    road_id = 20
    lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_id)[0]
    frenet_frame = MapUtils.get_lane_frenet_frame(lane_id)
    sub_segment = FrenetSubSegment(lane_id, 0, frenet_frame.s_max)
    reference_route = GeneralizedFrenetSerretFrame.build([frenet_frame], [sub_segment])

    dim_state = DriverInitiatedMotionState(logger)

    # too far from the stop sign
    lane_fstate = np.array([stop_bar.e_l_station - 7., 0, 0, 0, 0, 0])

    # pedal pressed
    pedal_position = create_pedal_position(time_in_sec=0, pedal_pos=0.1)
    dim_state.update_pedal_times(pedal_position)
    dim_state.update_state(pedal_position.s_Data.s_RecvTimestamp.timestamp_in_seconds, lane_id, lane_fstate, reference_route)
    assert dim_state.stop_bar_to_ignore() is None

    # pedal pressed for enough time
    pedal_position = create_pedal_position(time_in_sec=2, pedal_pos=0.1)
    dim_state.update_pedal_times(pedal_position)
    dim_state.update_state(pedal_position.s_Data.s_RecvTimestamp.timestamp_in_seconds, lane_id, lane_fstate, reference_route)
    assert dim_state.stop_bar_to_ignore() is None

    # near the stop sign
    lane_fstate = np.array([stop_bar.e_l_station - 2., 0, 0, 0, 0, 0])

    # pedal pressed
    pedal_position = create_pedal_position(time_in_sec=0, pedal_pos=0.1)
    dim_state.update_pedal_times(pedal_position)
    dim_state.update_state(pedal_position.s_Data.s_RecvTimestamp.timestamp_in_seconds, lane_id, lane_fstate, reference_route)
    assert dim_state.stop_bar_to_ignore() is None

    # pedal pressed for enough time
    pedal_position = create_pedal_position(time_in_sec=2, pedal_pos=0.1)
    dim_state.update_pedal_times(pedal_position)
    dim_state.update_state(pedal_position.s_Data.s_RecvTimestamp.timestamp_in_seconds, lane_id, lane_fstate, reference_route)
    assert dim_state.stop_bar_to_ignore() is not None

    # pedal released
    pedal_position = create_pedal_position(time_in_sec=3, pedal_pos=0.01)
    dim_state.update_pedal_times(pedal_position)
    dim_state.update_state(pedal_position.s_Data.s_RecvTimestamp.timestamp_in_seconds, lane_id, lane_fstate, reference_route)
    assert dim_state.stop_bar_to_ignore() is not None

    # crossed the stop sign
    lane_fstate = np.array([stop_bar.e_l_station + 5., 0, 0, 0, 0, 0])

    pedal_position = create_pedal_position(time_in_sec=4, pedal_pos=0.01)
    dim_state.update_pedal_times(pedal_position)
    dim_state.update_state(pedal_position.s_Data.s_RecvTimestamp.timestamp_in_seconds, lane_id, lane_fstate, reference_route)
    assert dim_state.stop_bar_to_ignore() is None

    # timeout of the DIM state
    pedal_position = create_pedal_position(time_in_sec=33, pedal_pos=0.01)
    dim_state.update_pedal_times(pedal_position)
    dim_state.update_state(pedal_position.s_Data.s_RecvTimestamp.timestamp_in_seconds, lane_id, lane_fstate, reference_route)
    assert dim_state.stop_bar_to_ignore() is None


def create_pedal_position(time_in_sec: float, pedal_pos: float):
    return PedalPosition(Header(0, Timestamp.from_seconds(time_in_sec), 0),
                         DataPedalPosition(0, pedal_pos, Timestamp.from_seconds(time_in_sec), True))
