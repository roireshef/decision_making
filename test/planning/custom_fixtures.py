import numpy as np
import pytest
from decision_making.src.global_constants import BEHAVIORAL_PLANNING_NAME_FOR_LOGGING, \
    TRAJECTORY_PLANNING_NAME_FOR_LOGGING, ROUTE_PLANNING_NAME_FOR_LOGGING, EGO_LENGTH, EGO_WIDTH, EGO_HEIGHT, \
    VELOCITY_LIMITS, LON_ACC_LIMITS, LAT_ACC_LIMITS, LON_JERK_COST_WEIGHT, LAT_JERK_COST_WEIGHT, \
    BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED, MAX_BACKWARD_HORIZON
from decision_making.src.messages.route_plan_message import RoutePlan, DataRoutePlan, RoutePlanLaneSegment
from decision_making.src.messages.scene_common_messages import Timestamp, Header
from decision_making.src.messages.scene_dynamic_message import SceneDynamic, DataSceneDynamic, HostLocalization, \
    HostHypothesis, ObjectLocalization, BoundingBoxSize, ObjectClassification, ObjectHypothesis, \
    ObjectTrackDynamicProperty
from decision_making.src.messages.scene_tcd_message import DataSceneTrafficControlDevices, SceneTrafficControlDevices
from decision_making.src.messages.trajectory_parameters import SigmoidFunctionParams, TrajectoryCostParams, \
    TrajectoryParams
from decision_making.src.messages.turn_signal_message import TurnSignal, DataTurnSignal, TurnSignalState
from decision_making.src.messages.visualization.behavioral_visualization_message import BehavioralVisualizationMsg
from decision_making.src.planning.behavioral.state.lane_change_state import LaneChangeState
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GeneralizedFrenetSerretFrame, \
    FrenetSubSegment
from decision_making.src.prediction.ego_aware_prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.scene.scene_static_model import SceneStaticModel
from decision_making.src.state.map_state import MapState
from decision_making.src.state.state import OccupancyState, ObjectSize, State, DynamicObject, EgoState
from decision_making.src.state.state import DynamicObjectsData
from decision_making.src.utils.map_utils import MapUtils
from decision_making.test.constants import LCM_PUB_SUB_MOCK_NAME_FOR_LOGGING
from decision_making.test.planning.behavioral.mock_behavioral_facade import BehavioralFacadeMock
from decision_making.test.planning.route.route_planner_mock import RoutePlannerMock
from decision_making.test.planning.trajectory.mock_trajectory_planning_facade import TrajectoryPlanningFacadeMock
from decision_making.test.pubsub.mock_pubsub import PubSubMock
from rte.python.logger.AV_logger import AV_Logger

UPDATED_TIMESTAMP_PARAM = 'updated_timestamp'
OLD_TIMESTAMP_PARAM = 'old_timestamp'


### MESSAGES ###

@pytest.fixture(scope='function')
def car_size():
    yield ObjectSize(length=3.0, width=2.0, height=1.2)


@pytest.fixture(scope='function')
def route_plan_1_2():
    yield RoutePlan(s_Header=Header(e_Cnt_SeqNum=1, s_Timestamp=Timestamp(0, 0), e_Cnt_version=1),
                    s_Data=DataRoutePlan(e_b_is_valid=True,
                                         e_Cnt_num_road_segments=2,
                                         a_i_road_segment_ids=np.array([1, 2]),
                                         a_Cnt_num_lane_segments=np.array([3, 3]),
                                         as_route_plan_lane_segments=[
                                             [RoutePlanLaneSegment(10, 0, 0), RoutePlanLaneSegment(11, 0, 0), RoutePlanLaneSegment(12, 0, 0)],
                                             [RoutePlanLaneSegment(20, 0, 0), RoutePlanLaneSegment(21, 0, 0), RoutePlanLaneSegment(22, 0, 0)]]))

@pytest.fixture(scope='function')
def route_plan_1_2_3():
    yield RoutePlan(s_Header=Header(e_Cnt_SeqNum=1, s_Timestamp=Timestamp(0, 0), e_Cnt_version=1),
                    s_Data=DataRoutePlan(e_b_is_valid=True,
                                         e_Cnt_num_road_segments=3,
                                         a_i_road_segment_ids=np.array([1, 2, 3]),
                                         a_Cnt_num_lane_segments=np.array([3, 3, 3]),
                                         as_route_plan_lane_segments=[
                                             [RoutePlanLaneSegment(10, 0, 0), RoutePlanLaneSegment(11, 0, 0), RoutePlanLaneSegment(12, 0, 0)],
                                             [RoutePlanLaneSegment(20, 0, 0), RoutePlanLaneSegment(21, 0, 0), RoutePlanLaneSegment(22, 0, 0)],
                                             [RoutePlanLaneSegment(30, 0, 0), RoutePlanLaneSegment(31, 0, 0), RoutePlanLaneSegment(32, 0, 0)]]))

@pytest.fixture(scope='function')
def route_plan_left_lane_ends(route_plan_1_2):
    # Delete left lane
    route_plan_1_2.s_Data.a_Cnt_num_lane_segments[1] -= 1
    del route_plan_1_2.s_Data.as_route_plan_lane_segments[1][-1]

    yield route_plan_1_2

@pytest.fixture(scope='function')
def route_plan_right_lane_ends(route_plan_1_2):
    # Delete right lane
    route_plan_1_2.s_Data.a_Cnt_num_lane_segments[1] -= 1
    del route_plan_1_2.s_Data.as_route_plan_lane_segments[1][0]

    yield route_plan_1_2

@pytest.fixture(scope='function')
def route_plan_lane_split_on_right(route_plan_1_2):
    # Delete right lane in road segment 1
    route_plan_1_2.s_Data.a_Cnt_num_lane_segments[0] -= 1
    del route_plan_1_2.s_Data.as_route_plan_lane_segments[0][0]

    yield route_plan_1_2

@pytest.fixture(scope='function')
def route_plan_lane_split_on_left(route_plan_1_2):
    # Delete left lane in road segment 1
    route_plan_1_2.s_Data.a_Cnt_num_lane_segments[0] -= 1
    del route_plan_1_2.s_Data.as_route_plan_lane_segments[0][2]

    yield route_plan_1_2

@pytest.fixture(scope='function')
def route_plan_lane_split_on_left_and_right(route_plan_1_2):
    # Delete right lane in road segment 1
    route_plan_1_2.s_Data.a_Cnt_num_lane_segments[0] -= 1
    del route_plan_1_2.s_Data.as_route_plan_lane_segments[0][0]
    # Delete left lane in road segment 1
    route_plan_1_2.s_Data.a_Cnt_num_lane_segments[0] -= 1
    del route_plan_1_2.s_Data.as_route_plan_lane_segments[0][1]

    yield route_plan_1_2

@pytest.fixture(scope='function')
def dynamic_objects_not_on_road():

    obj_id = 1
    confidence = 1.0
    bbox = BoundingBoxSize(2, 2, 2)

    location_x = 17
    location_y = 17
    yaw = 0.982
    glob_v_x = 2
    glob_v_y = 3

    # convert velocity from map coordinates to relative to its own yaw
    v_x = np.cos(yaw) * glob_v_x + np.sin(yaw) * glob_v_y
    v_y = -np.sin(yaw) * glob_v_x + np.cos(yaw) * glob_v_y
    total_v = np.linalg.norm([v_x, v_y])
    cartesian_state = np.array([location_x, location_y, yaw, total_v, 0, 0])

    objects_localization = [ObjectLocalization(e_Cnt_object_id=obj_id,
                                               e_e_object_type=ObjectClassification.CeSYS_e_ObjectClassification_Car,
                                               s_bounding_box=bbox, a_cartesian_pose=cartesian_state,
                                               e_Cnt_obj_hypothesis_count=1,
                                               as_object_hypothesis=[ObjectHypothesis(e_r_probability=confidence,
                                                                                      e_i_road_segment_id=0,
                                                                                      e_i_lane_segment_id=0,
                                                                                      e_e_dynamic_status=ObjectTrackDynamicProperty.CeSYS_e_ObjectTrackDynProp_Unknown,
                                                                                      e_Pct_location_uncertainty_x=0,
                                                                                      e_Pct_location_uncertainty_y=0,
                                                                                      e_Pct_location_uncertainty_yaw=0,
                                                                                      e_i_host_lane_frenet_id=0,
                                                                                      a_lane_frenet_pose=np.zeros(6),
                                                                                      a_host_lane_frenet_pose=np.zeros(6), e_b_off_lane=True)])]
    objects = DynamicObjectsData(num_objects=1, objects_localization=objects_localization, timestamp=3)
    yield objects


@pytest.fixture(scope='function')
def dynamic_objects_negative_velocity():

    obj_id = 1
    confidence = 1.0
    bbox = BoundingBoxSize(2, 2, 2)

    location_x = 5
    location_y = 1
    yaw = 3.14
    glob_v_x = 1
    glob_v_y = 0

    # convert velocity from map coordinates to relative to its own yaw
    v_x = np.cos(yaw) * glob_v_x + np.sin(yaw) * glob_v_y
    v_y = -np.sin(yaw) * glob_v_x + np.cos(yaw) * glob_v_y
    total_v = np.linalg.norm([v_x, v_y])
    cartesian_state = np.array([location_x, location_y, yaw, total_v, 0, 0])

    objects_localization = [ObjectLocalization(e_Cnt_object_id=obj_id,
                                               e_e_object_type=ObjectClassification.CeSYS_e_ObjectClassification_Car,
                                               s_bounding_box=bbox, a_cartesian_pose=cartesian_state,
                                               e_Cnt_obj_hypothesis_count=1,
                                               as_object_hypothesis=[ObjectHypothesis(e_r_probability=confidence,
                                                                                      e_i_road_segment_id=0,
                                                                                      e_i_lane_segment_id=0,
                                                                                      e_e_dynamic_status=ObjectTrackDynamicProperty.CeSYS_e_ObjectTrackDynProp_Unknown,
                                                                                      e_Pct_location_uncertainty_x=0,
                                                                                      e_Pct_location_uncertainty_y=0,
                                                                                      e_Pct_location_uncertainty_yaw=0,
                                                                                      e_i_host_lane_frenet_id=0,
                                                                                      a_lane_frenet_pose=None,
                                                                                      a_host_lane_frenet_pose=None)])]
    objects = DynamicObjectsData(num_objects=1, objects_localization=objects_localization, timestamp=3)
    yield objects


@pytest.fixture(scope='function')
def state(scene_static_short_testable):
    SceneStaticModel.get_instance().set_scene_static(scene_static_short_testable)

    occupancy_state = OccupancyState(0, np.array([]), np.array([]))
    v_x = 2.0
    v_y = 2.0
    v = np.linalg.norm([v_x, v_y])
    dyn1 = DynamicObject(obj_id=1, timestamp=34, cartesian_state=np.array([0.5, 0.1, np.pi / 8.0, v, 0.0, 0.0]),
                         map_state=MapState(lane_fstate=np.array([0.5, 2.61312593, 0., 0.1, 1.0823922, 0.]), lane_id=11),
                         size=ObjectSize(1, 1, 1), confidence=1.0, off_map=False)
    dyn2 = DynamicObject(obj_id=2, timestamp=35, cartesian_state=np.array([10.0, 0.0, np.pi / 8.0, v, 0.0, 0.0]),
                         map_state=MapState(lane_fstate=np.array([10., 2.61312593, 0., 0., 1.0823922, 0.]), lane_id=11),
                         size=ObjectSize(1, 1, 1), confidence=1.0, off_map=False)

    dynamic_objects = [dyn1, dyn2]
    size = ObjectSize(EGO_LENGTH, EGO_WIDTH, EGO_HEIGHT)
    ego_state = EgoState(obj_id=0, timestamp=0, cartesian_state=np.array([1, 0, 0, 1.0, 0.0, 0]),
                         map_state=MapState(lane_fstate=np.array([1., 1., 0., 0., 0., 0.]), lane_id=11),
                         size=size, confidence=0, off_map=False)
    yield State(False, occupancy_state, dynamic_objects, ego_state)


@pytest.fixture(scope='function')
def state_with_old_object(request) -> State:
    """
    :return: a state object with an old object
    """
    updated_timestamp = request.param[UPDATED_TIMESTAMP_PARAM]
    old_timestamp = request.param[OLD_TIMESTAMP_PARAM]
    occupancy_state = OccupancyState(0, np.array([]), np.array([]))
    v_x = 2.0
    v_y = 2.0
    v = np.linalg.norm([v_x, v_y])
    dyn1 = DynamicObject.create_from_cartesian_state(obj_id=1, timestamp=updated_timestamp,
                                                     cartesian_state=np.array([0.1, 0.1, np.pi / 8.0, v, 0.0, 0.0]),
                                                     size=ObjectSize(1, 1, 1), confidence=1.0)
    dyn2 = DynamicObject.create_from_cartesian_state(obj_id=2, timestamp=old_timestamp, cartesian_state=np.array([10.0, 0.0, np.pi / 8.0, v, 0.0, 0.0]),
                                                     size=ObjectSize(1, 1, 1), confidence= 1.0)
    dynamic_objects = [dyn1, dyn2]
    size = ObjectSize(EGO_LENGTH, EGO_WIDTH, EGO_HEIGHT)

    ego_state = EgoState.create_from_cartesian_state(obj_id=1, timestamp=old_timestamp, cartesian_state=np.array([1, 0, 0, 1.0, 0.0, 0]),
                                                     size=size, confidence=0)

    yield State(False, occupancy_state, dynamic_objects, ego_state)


@pytest.fixture(scope='function')
def scene_dynamic(scene_static_short_testable) -> SceneDynamic:

    SceneStaticModel.get_instance().set_scene_static(scene_static_short_testable)

    lane_id = 11
    road_id = 1
    fstate = np.array([MAX_BACKWARD_HORIZON, 1., 0., 0., 0., 0.])
    host_hypotheses = [HostHypothesis(road_id, lane_id, fstate, False)]

    frenet = MapUtils.get_lane_frenet_frame(lane_id)
    cstate = frenet.fstate_to_cstate(fstate)
    ego_localization = HostLocalization(cstate, 1, host_hypotheses)

    timestamp = Timestamp.from_seconds(5.0)
    header = Header(0, timestamp, 0)
    data = DataSceneDynamic(True, timestamp, timestamp, 0, [], ego_localization)
    scene_dynamic = SceneDynamic(s_Header=header, s_Data=data)

    yield scene_dynamic


@pytest.fixture(scope='function')
def tcd_status() -> SceneTrafficControlDevices:

    timestamp = Timestamp.from_seconds(5.0)
    tcd_status_data = DataSceneTrafficControlDevices(s_RecvTimestamp=timestamp, s_ComputeTimestamp=timestamp, as_dynamic_traffic_control_device_status={})
    header = Header(0, timestamp, 0)
    tcd_status = SceneTrafficControlDevices(s_Header=header, s_Data=tcd_status_data)

    yield tcd_status


@pytest.fixture(scope='function')
def scene_dynamic_fix_single_host_hypothesis(scene_static_pg_split):

    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)

    lane_id = 200
    road_id = 20
    fstate = np.array([10, 5, 0, 0, 0, 0])
    host_hypotheses = [HostHypothesis(road_id, lane_id, fstate, False)]

    frenet = MapUtils.get_lane_frenet_frame(lane_id)
    cstate = frenet.fstate_to_cstate(fstate)
    ego_localization = HostLocalization(cstate, 1, host_hypotheses)

    timestamp = Timestamp.from_seconds(5.0)
    header = Header(0, timestamp, 0)
    data = DataSceneDynamic(True, timestamp, timestamp, 0, [], ego_localization)
    scene_dynamic = SceneDynamic(s_Header=header, s_Data=data)

    yield scene_dynamic


@pytest.fixture(scope='function')
def scene_dynamic_fix_two_host_hypotheses(scene_static_pg_split):

    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)

    lane_id1 = 200
    road_id1 = 20
    fstate1 = np.array([50, 5, 0, 1.8, 0, 0])
    host_hyp1 = HostHypothesis(road_id1, lane_id1, fstate1, False)

    lane_id2 = 201
    road_id2 = 20
    fstate2 = np.array([50, 5, 0, -1.8, 0, 0])
    host_hyp2 = HostHypothesis(road_id2, lane_id2, fstate2, False)

    host_hypotheses = [host_hyp1, host_hyp2]

    frenet = MapUtils.get_lane_frenet_frame(lane_id1)
    cstate = frenet.fstate_to_cstate(fstate1)
    ego_localization = HostLocalization(cstate, 2, host_hypotheses)

    timestamp = Timestamp.from_seconds(5.0)
    header = Header(0, timestamp, 0)
    data = DataSceneDynamic(True, timestamp, timestamp, 0, [], ego_localization)
    scene_dynamic = SceneDynamic(s_Header=header, s_Data=data)

    yield scene_dynamic


@pytest.fixture(scope='function')
def scene_dynamic_fix_three_host_hypotheses(scene_static_oval_with_splits):

    SceneStaticModel.get_instance().set_scene_static(scene_static_oval_with_splits)

    lane_id1 = 2244100
    road_id1 = MapUtils.get_road_segment_id_from_lane_id(lane_id1)
    fstate1 = np.array([MapUtils.get_lane_length(2244100), 10, 0, 0, 0, 0])
    host_hyp1 = HostHypothesis(road_id1, lane_id1, fstate1, False)

    lane_id2 = 19670532
    road_id2 = MapUtils.get_road_segment_id_from_lane_id(lane_id2)
    fstate2 = np.array([0, 10, 0, 0, 0, 0])
    host_hyp2 = HostHypothesis(road_id2, lane_id2, fstate2, False)

    lane_id3 = 19670533
    road_id3 = MapUtils.get_road_segment_id_from_lane_id(lane_id3)
    fstate3 = np.array([0, 10, 0, 0, 0, 0])
    host_hyp3 = HostHypothesis(road_id3, lane_id3, fstate3, False)

    host_hypotheses = [host_hyp1, host_hyp2, host_hyp3]

    frenet = MapUtils.get_lane_frenet_frame(lane_id1)
    cstate = frenet.fstate_to_cstate(fstate1)
    ego_localization = HostLocalization(cstate, 3, host_hypotheses)

    timestamp = Timestamp.from_seconds(5.0)
    header = Header(0, timestamp, 0)
    data = DataSceneDynamic(True, timestamp, timestamp, 0, [], ego_localization)
    scene_dynamic = SceneDynamic(s_Header=header, s_Data=data)

    yield scene_dynamic


@pytest.fixture(scope='function')
def ego_state_fix():
    size = ObjectSize(0, 0, 0)

    ego_state = EgoState.create_from_cartesian_state(obj_id=0, timestamp=5, cartesian_state=np.array([0, 0, 0, 1.0, 0.0, 0]),
                                                     size=size, confidence=0)
    yield ego_state


@pytest.fixture(scope='function')
def dyn_obj_on_road():
    size = ObjectSize(0, 0, 0)
    dyn_obj = DynamicObject.create_from_cartesian_state(obj_id=0, timestamp=5, cartesian_state=np.array([5.0, 1.0, 0, 1.0, 0.0, 0]),
                                                        size=size, confidence=0)
    yield dyn_obj


@pytest.fixture(scope='function')
def dyn_obj_outside_road():
    size = ObjectSize(0, 0, 0)
    dyn_obj = DynamicObject.create_from_cartesian_state(obj_id=0, timestamp=5, cartesian_state=np.array([5.0, -10.0, 0, 1.0, 0.0, 0]),
                                                        size=size, confidence=0)
    yield dyn_obj


@pytest.fixture(scope='function')
def trajectory_params():
    ref_points = np.array([[x, -2.0] for x in range(0, 200)])
    frenet = FrenetSerret2DFrame.fit(ref_points)
    ref_route = GeneralizedFrenetSerretFrame.build([frenet], [FrenetSubSegment(0, 0, frenet.s_max)])
    target_state = np.array([15.0, -2.0, 0.0, 1, 0.0, 0.0])
    mock_sigmoid = SigmoidFunctionParams(1.0, 2.0, 3.0)
    trajectory_cost_params = TrajectoryCostParams(mock_sigmoid, mock_sigmoid, mock_sigmoid, mock_sigmoid,
                                                  mock_sigmoid, mock_sigmoid, mock_sigmoid, mock_sigmoid,
                                                  mock_sigmoid, 3.0, LON_JERK_COST_WEIGHT, LAT_JERK_COST_WEIGHT,
                                                  VELOCITY_LIMITS, LON_ACC_LIMITS, LAT_ACC_LIMITS,
                                                  BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED)
    yield TrajectoryParams(reference_route=ref_route, target_state=target_state,
                           cost_params=trajectory_cost_params, target_time=16, trajectory_end_time=16,
                           strategy=TrajectoryPlanningStrategy.HIGHWAY,
                           bp_time=0)


### VIZ MESSAGES ###

@pytest.fixture(scope='function')
def behavioral_visualization_msg(trajectory_params):
    yield BehavioralVisualizationMsg(trajectory_params.reference_route.points)


### MODULES/INFRA ###

@pytest.fixture(scope='function')
def pubsub():
    yield PubSubMock(logger=AV_Logger.get_logger(LCM_PUB_SUB_MOCK_NAME_FOR_LOGGING))


@pytest.fixture(scope='function')
def route_planner_facade(state, pubsub, route_plan_1_2):
    logger = AV_Logger.get_logger(ROUTE_PLANNING_NAME_FOR_LOGGING)

    route_plan_mock = RoutePlannerMock(pubsub, logger, route_plan_1_2)
    route_plan_mock.start()
    yield route_plan_mock
    route_plan_mock.stop()


@pytest.fixture(scope='function')
def behavioral_facade(pubsub, trajectory_params, behavioral_visualization_msg):
    logger = AV_Logger.get_logger(BEHAVIORAL_PLANNING_NAME_FOR_LOGGING)

    behavioral_module = BehavioralFacadeMock(pubsub=pubsub, logger=logger, trajectory_params=trajectory_params,
                                             visualization_msg=behavioral_visualization_msg, trigger_pos=None)

    behavioral_module.start()
    yield behavioral_module
    behavioral_module.stop()


@pytest.fixture(scope='function')
def trajectory_planner_facade(pubsub, trajectory, trajectory_visualization_msg):
    logger = AV_Logger.get_logger(TRAJECTORY_PLANNING_NAME_FOR_LOGGING)

    trajectory_planning_module = TrajectoryPlanningFacadeMock(pubsub=pubsub, logger=logger,
                                                              trajectory_msg=trajectory,
                                                              visualization_msg=trajectory_visualization_msg)

    trajectory_planning_module.start()
    yield trajectory_planning_module
    trajectory_planning_module.stop()


@pytest.fixture(scope='function')
def predictor():
    logger = AV_Logger.get_logger("PREDICTOR_TEST_LOGGER")
    yield RoadFollowingPredictor(logger)


@pytest.fixture(scope='function')
def turn_signal() -> TurnSignal:
    timestamp = Timestamp.from_seconds(5.0)
    header = Header(0, timestamp, 0)
    data = DataTurnSignal(True, timestamp, timestamp, TurnSignalState.CeSYS_e_Off, np.array([]))
    turn_signal = TurnSignal(s_Header=header, s_Data=data)

    yield turn_signal


@pytest.fixture(scope='session')
def lane_change_state() -> LaneChangeState:
    yield LaneChangeState(None, np.array([]), False, None)