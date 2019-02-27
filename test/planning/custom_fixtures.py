import numpy as np
import pytest
from decision_making.src.state.map_state import MapState

from decision_making.src.global_constants import STATE_MODULE_NAME_FOR_LOGGING, BEHAVIORAL_PLANNING_NAME_FOR_LOGGING, \
    NAVIGATION_PLANNING_NAME_FOR_LOGGING, TRAJECTORY_PLANNING_NAME_FOR_LOGGING, EGO_LENGTH, EGO_WIDTH, EGO_HEIGHT, \
    VELOCITY_LIMITS, LON_ACC_LIMITS, LAT_ACC_LIMITS, LON_JERK_COST_WEIGHT, LAT_JERK_COST_WEIGHT
from decision_making.src.scene.scene_static_model import SceneStaticModel
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.messages.scene_common_messages import Timestamp, Header, MapOrigin
from decision_making.src.messages.scene_dynamic_message import SceneDynamic, DataSceneDynamic, HostLocalization, \
    ObjectLocalization, BoundingBoxSize, ObjectClassification, ObjectHypothesis, ObjectTrackDynamicProperty
from decision_making.src.messages.scene_static_message import DynamicStatus, TrafficSignalState
from decision_making.src.messages.trajectory_parameters import SigmoidFunctionParams, TrajectoryCostParams, \
    TrajectoryParams
from decision_making.src.messages.trajectory_plan_message import TrajectoryPlan
from decision_making.src.messages.visualization.behavioral_visualization_message import BehavioralVisualizationMsg
from decision_making.src.messages.visualization.trajectory_visualization_message import TrajectoryVisualizationMsg
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GeneralizedFrenetSerretFrame, \
    FrenetSubSegment
from decision_making.src.prediction.ego_aware_prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.state.state import OccupancyState, ObjectSize, State, DynamicObject, EgoState
from decision_making.src.state.state_module import DynamicObjectsData
from decision_making.src.utils.map_utils import MapUtils
from decision_making.test.constants import LCM_PUB_SUB_MOCK_NAME_FOR_LOGGING
from decision_making.test.messages.static_scene_fixture import scene_static
from decision_making.test.planning.behavioral.mock_behavioral_facade import BehavioralFacadeMock
from decision_making.test.planning.navigation.mock_navigation_facade import NavigationFacadeMock
from decision_making.test.planning.trajectory.mock_trajectory_planning_facade import TrajectoryPlanningFacadeMock
from decision_making.test.pubsub.mock_pubsub import PubSubMock
from decision_making.test.state.mock_state_module import StateModuleMock
from rte.ctm.pythonwrappers.src.FrenetSerret2DFrame import FrenetSerret2DFrame
from rte.python.logger.AV_logger import AV_Logger

from decision_making.test.messages.static_scene_fixture import create_scene_static_from_map_api

from mapping.test.model.testable_map_fixtures import ROAD_WIDTH, MAP_INFLATION_FACTOR, navigation_fixture,\
    short_testable_map_api, testable_map_api

UPDATED_TIMESTAMP_PARAM = 'updated_timestamp'
OLD_TIMESTAMP_PARAM = 'old_timestamp'


### MESSAGES ###

@pytest.fixture(scope='function')
def car_size():
    yield ObjectSize(length=3.0, width=2.0, height=1.2)


@pytest.fixture(scope='function')
def navigation_plan():
    yield NavigationPlanMsg(np.array([1, 2]))


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
                                               s_bounding_box=bbox, e_Cnt_obj_hypothesis_count=1,
                                               as_object_hypothesis=[ObjectHypothesis(e_r_probability=confidence,
                                                                                      e_i_lane_segment_id=0,
                                                                                      e_e_dynamic_status=ObjectTrackDynamicProperty.CeSYS_e_ObjectTrackDynProp_Unknown,
                                                                                      e_Pct_location_uncertainty_x=0,
                                                                                      e_Pct_location_uncertainty_y=0,
                                                                                      e_Pct_location_uncertainty_yaw=0,
                                                                                      e_i_host_lane_frenet_id=0,
                                                                                      a_cartesian_pose=cartesian_state,
                                                                                      a_lane_frenet_pose=np.zeros(6),
                                                                                      a_host_lane_frenet_pose=np.zeros(6))])]
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
                                               s_bounding_box=bbox, e_Cnt_obj_hypothesis_count=1,
                                               as_object_hypothesis=[ObjectHypothesis(e_r_probability=confidence,
                                                                                      e_i_lane_segment_id=0,
                                                                                      e_e_dynamic_status=ObjectTrackDynamicProperty.CeSYS_e_ObjectTrackDynProp_Unknown,
                                                                                      e_Pct_location_uncertainty_x=0,
                                                                                      e_Pct_location_uncertainty_y=0,
                                                                                      e_Pct_location_uncertainty_yaw=0,
                                                                                      e_i_host_lane_frenet_id=0,
                                                                                      a_cartesian_pose=cartesian_state,
                                                                                      a_lane_frenet_pose=None,
                                                                                      a_host_lane_frenet_pose=None)])]
    objects = DynamicObjectsData(num_objects=1, objects_localization=objects_localization, timestamp=3)
    yield objects


@pytest.fixture(scope='function')
def state(short_testable_map_api):
    short_scene_static = create_scene_static_from_map_api(short_testable_map_api)
    SceneStaticModel.get_instance().set_scene_static(short_scene_static)

    occupancy_state = OccupancyState(0, np.array([]), np.array([]))
    v_x = 2.0
    v_y = 2.0
    v = np.linalg.norm([v_x, v_y])
    dyn1 = DynamicObject(obj_id=1, timestamp=34, cartesian_state=np.array([0.5, 0.1, np.pi / 8.0, v, 0.0, 0.0]),
                         map_state=MapState(lane_fstate=np.array([0.5, 2.61312593, 0., 0.1, 1.0823922, 0.]), lane_id=11),
                         map_state_on_host_lane=MapState(lane_fstate=np.array([0.5, 2.61312593, 0., 0.1, 1.0823922, 0.]), lane_id=11),
                         size=ObjectSize(1, 1, 1), confidence=1.0)
    dyn2 = DynamicObject(obj_id=2, timestamp=35, cartesian_state=np.array([10.0, 0.0, np.pi / 8.0, v, 0.0, 0.0]),
                         map_state=MapState(lane_fstate=np.array([10., 2.61312593, 0., 0., 1.0823922, 0.]), lane_id=11),
                         map_state_on_host_lane=MapState(lane_fstate=np.array([0.5, 2.61312593, 0., 0.1, 1.0823922, 0.]), lane_id=11),
                         size=ObjectSize(1, 1, 1), confidence=1.0)
    dyn1.map_state
    dyn2.map_state

    dynamic_objects = [dyn1, dyn2]
    size = ObjectSize(EGO_LENGTH, EGO_WIDTH, EGO_HEIGHT)
    ego_state = EgoState(obj_id=0, timestamp=0, cartesian_state=np.array([1, 0, 0, 1.0, 0.0, 0]),
                         map_state=MapState(lane_fstate=np.array([1., 1., 0., 0., 0., 0.]), lane_id=11),
                         map_state_on_host_lane=MapState(lane_fstate=np.array([1., 1., 0., 0., 0., 0.]), lane_id=11),
                         size=size, confidence=0)
    yield State(occupancy_state, dynamic_objects, ego_state)


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

    yield State(occupancy_state, dynamic_objects, ego_state)


@pytest.fixture(scope='function')
def scene_dynamic_fix():

    SceneStaticModel.get_instance().set_scene_static(scene_static())
    lane_id = 200
    cstate = np.array([1100, 7, 0, 1.0, 0.0, 0])

    frenet = MapUtils.get_lane_frenet_frame(lane_id)
    fstate = frenet.cstate_to_fstate(cstate)

    timestamp = Timestamp.from_seconds(5.0)
    ego_localization = HostLocalization(lane_id, 0, cstate, fstate)
    header = Header(0, timestamp, 0)
    data = DataSceneDynamic(True, timestamp, timestamp, 0, [], ego_localization)
    map_origin = MapOrigin(0.0, 0.0, 0.0, timestamp)
    scene_dynamic = SceneDynamic(s_Header=header, s_Data=data, s_MapOrigin=map_origin)

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
                                                  VELOCITY_LIMITS, LON_ACC_LIMITS, LAT_ACC_LIMITS)
    yield TrajectoryParams(reference_route=ref_route, target_state=target_state,
                           cost_params=trajectory_cost_params, time=16,
                           strategy=TrajectoryPlanningStrategy.HIGHWAY,
                           bp_time=0)


@pytest.fixture(scope='function')
def trajectory():
    chosen_trajectory = np.array(
        [[1.0, 0.0, 0.0, 0.0], [2.0, -0.33, 0.0, 0.0], [3.0, -0.66, 0.0, 0.0], [4.0, -1.0, 0.0, 0.0],
         [5.0, -1.33, 0.0, 0.0], [6.0, -1.66, 0.0, 0.0], [7.0, -2.0, 0.0, 0.0], [8.0, -2.0, 0.0, 0.0],
         [9.0, -2.0, 0.0, 0.0], [10.0, -2.0, 0.0, 0.0], [11.0, -2.0, 0.0, 0.0]])
    yield TrajectoryPlan(timestamp=0, trajectory=chosen_trajectory, current_speed=5.0)


### VIZ MESSAGES ###

@pytest.fixture(scope='function')
def behavioral_visualization_msg(trajectory_params):
    yield BehavioralVisualizationMsg(trajectory_params.reference_route.points)


@pytest.fixture(scope='function')
def trajectory_visualization_msg(state, trajectory):
    yield TrajectoryVisualizationMsg(reference_route=trajectory.trajectory,
                                     trajectories=np.array([trajectory.trajectory]),
                                     costs=np.array([0]),
                                     state=state,
                                     predicted_states=[state],
                                     plan_time=2.0)


### MODULES/INFRA ###

@pytest.fixture(scope='function')
def pubsub():
    yield PubSubMock(logger=AV_Logger.get_logger(LCM_PUB_SUB_MOCK_NAME_FOR_LOGGING))


@pytest.fixture(scope='function')
def state_module(state, pubsub):
    logger = AV_Logger.get_logger(STATE_MODULE_NAME_FOR_LOGGING)

    state_mock = StateModuleMock(pubsub, logger, state)
    state_mock.start()
    yield state_mock
    state_mock.stop()


@pytest.fixture(scope='function')
def behavioral_facade(pubsub, trajectory_params, behavioral_visualization_msg):
    logger = AV_Logger.get_logger(BEHAVIORAL_PLANNING_NAME_FOR_LOGGING)

    behavioral_module = BehavioralFacadeMock(pubsub=pubsub, logger=logger, trajectory_params=trajectory_params,
                                             visualization_msg=behavioral_visualization_msg, trigger_pos=None)

    behavioral_module.start()
    yield behavioral_module
    behavioral_module.stop()


@pytest.fixture(scope='function')
def navigation_facade(pubsub, navigation_plan):
    logger = AV_Logger.get_logger(NAVIGATION_PLANNING_NAME_FOR_LOGGING)

    navigation_module = NavigationFacadeMock(pubsub=pubsub, logger=logger, navigation_plan_msg=navigation_plan)

    navigation_module.start()
    yield navigation_module
    navigation_module.stop()


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
