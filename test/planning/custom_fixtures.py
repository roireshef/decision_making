import pytest
import numpy as np

from decision_making.src.global_constants import STATE_MODULE_NAME_FOR_LOGGING, BEHAVIORAL_PLANNING_NAME_FOR_LOGGING, \
    NAVIGATION_PLANNING_NAME_FOR_LOGGING, TRAJECTORY_PLANNING_NAME_FOR_LOGGING, EGO_LENGTH, EGO_WIDTH, EGO_HEIGHT, \
    VELOCITY_LIMITS, LON_ACC_LIMITS, LAT_ACC_LIMITS, LON_JERK_COST_WEIGHT, LAT_JERK_COST_WEIGHT, TIMESTAMP_RESOLUTION_IN_SEC
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.messages.trajectory_parameters import SigmoidFunctionParams, TrajectoryCostParams, \
    TrajectoryParams
from decision_making.src.messages.trajectory_plan_message import TrajectoryPlan
from decision_making.src.messages.visualization.behavioral_visualization_message import BehavioralVisualizationMsg
from decision_making.src.messages.visualization.trajectory_visualization_message import TrajectoryVisualizationMsg
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.prediction.ego_aware_prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.state.state import OccupancyState, ObjectSize, State, DynamicObject, EgoState
from decision_making.test.pubsub.mock_pubsub import PubSubMock
from decision_making.test.planning.behavioral.mock_behavioral_facade import BehavioralFacadeMock
from decision_making.test.planning.navigation.mock_navigation_facade import NavigationFacadeMock
from decision_making.test.planning.trajectory.mock_trajectory_planning_facade import TrajectoryPlanningFacadeMock
from decision_making.test.state.mock_state_module import StateModuleMock
from common_data.interface.py.idl_generated_files.Rte_Types import LcmPerceivedDynamicObjectList
from common_data.interface.py.idl_generated_files.Rte_Types.sub_structures.LcmPerceivedDynamicObject import LcmPerceivedDynamicObject
from common_data.interface.py.idl_generated_files.Rte_Types.sub_structures.LcmObjectLocation import LcmObjectLocation
from common_data.interface.py.idl_generated_files.Rte_Types.sub_structures.LcmObjectBbox import LcmObjectBbox
from common_data.interface.py.idl_generated_files.Rte_Types.sub_structures.LcmObjectVelocity import LcmObjectVelocity
from common_data.interface.py.idl_generated_files.Rte_Types.sub_structures.LcmObjectTrackingStatus import LcmObjectTrackingStatus

from rte.python.logger.AV_logger import AV_Logger
from decision_making.test.constants import LCM_PUB_SUB_MOCK_NAME_FOR_LOGGING

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
def dynamic_objects_in_fov():
    objects = LcmPerceivedDynamicObjectList()

    objects.timestamp = 1

    dyn_obj = LcmPerceivedDynamicObject()
    dyn_obj.id = 1

    dyn_obj.location = LcmObjectLocation()
    dyn_obj.location.x = 5
    dyn_obj.location.y = 1
    dyn_obj.location.confidence = 1.0

    dyn_obj.bbox = LcmObjectBbox()

    dyn_obj.bbox.yaw = 1.107
    dyn_obj.bbox.length = 2
    dyn_obj.bbox.width = 2
    dyn_obj.bbox.height = 2

    dyn_obj.velocity = LcmObjectVelocity()

    dyn_obj.velocity.v_x = 1
    dyn_obj.velocity.v_y = 2
    dyn_obj.velocity.omega_yaw = 0

    dyn_obj.tracking_status = LcmObjectTrackingStatus()
    dyn_obj.tracking_status.in_fov = True
    dyn_obj.tracking_status.is_predicted = False

    objects.num_objects = 1
    objects.dynamic_objects[0] = dyn_obj

    yield objects


@pytest.fixture(scope='function')
def dynamic_objects_not_in_fov():
    objects = LcmPerceivedDynamicObjectList()

    objects.timestamp = 3

    dyn_obj = LcmPerceivedDynamicObject()
    dyn_obj.id = 1

    dyn_obj.location = LcmObjectLocation()
    dyn_obj.location.x = 5
    dyn_obj.location.y = 1
    dyn_obj.location.confidence = 1.0

    dyn_obj.bbox = LcmObjectBbox()

    dyn_obj.bbox.yaw = 0.982
    dyn_obj.bbox.length = 2
    dyn_obj.bbox.width = 2
    dyn_obj.bbox.height = 2

    dyn_obj.velocity = LcmObjectVelocity()

    dyn_obj.velocity.v_x = 2
    dyn_obj.velocity.v_y = 3
    dyn_obj.velocity.omega_yaw = 0

    dyn_obj.tracking_status = LcmObjectTrackingStatus()
    dyn_obj.tracking_status.in_fov = False
    dyn_obj.tracking_status.is_predicted = False

    objects.num_objects = 1
    objects.dynamic_objects[0] = dyn_obj

    yield objects


@pytest.fixture(scope='function')
def dynamic_objects_not_on_road():
    objects = LcmPerceivedDynamicObjectList()

    objects.timestamp = 3

    dyn_obj = LcmPerceivedDynamicObject()
    dyn_obj.id = 1

    dyn_obj.location = LcmObjectLocation()
    dyn_obj.location.x = 17
    dyn_obj.location.y = 17
    dyn_obj.location.confidence = 1.0

    dyn_obj.bbox = LcmObjectBbox()

    dyn_obj.bbox.yaw = 0.982
    dyn_obj.bbox.length = 2
    dyn_obj.bbox.width = 2
    dyn_obj.bbox.height = 2

    dyn_obj.velocity = LcmObjectVelocity()

    dyn_obj.velocity.v_x = 2
    dyn_obj.velocity.v_y = 3
    dyn_obj.velocity.omega_yaw = 0

    dyn_obj.tracking_status = LcmObjectTrackingStatus()
    dyn_obj.tracking_status.in_fov = True
    dyn_obj.tracking_status.is_predicted = False

    objects.num_objects = 1
    objects.dynamic_objects[0] = dyn_obj

    yield objects


@pytest.fixture(scope='function')
def dynamic_objects_negative_velocity():
    objects = LcmPerceivedDynamicObjectList()

    objects.timestamp = 3

    dyn_obj = LcmPerceivedDynamicObject()
    dyn_obj.id = 1

    dyn_obj.location = LcmObjectLocation()
    dyn_obj.location.x = 5
    dyn_obj.location.y = 1
    dyn_obj.location.confidence = 1.0

    dyn_obj.bbox = LcmObjectBbox()

    dyn_obj.bbox.yaw = 3.14
    dyn_obj.bbox.length = 2
    dyn_obj.bbox.width = 2
    dyn_obj.bbox.height = 2

    dyn_obj.velocity = LcmObjectVelocity()

    dyn_obj.velocity.v_x = 1
    dyn_obj.velocity.v_y = 0
    dyn_obj.velocity.omega_yaw = 0

    dyn_obj.tracking_status = LcmObjectTrackingStatus()
    dyn_obj.tracking_status.in_fov = True
    dyn_obj.tracking_status.is_predicted = False

    objects.num_objects = 1
    objects.dynamic_objects[0] = dyn_obj

    yield objects


@pytest.fixture(scope='function')
def state():
    occupancy_state = OccupancyState(0, np.array([]), np.array([]))
    v_x = 2.0
    v_y = 2.0
    v = np.linalg.norm([v_x, v_y])
    dyn1 = DynamicObject.create_from_cartesian_state(obj_id=1, timestamp=34, cartesian_state=np.array([0.5, 0.1, np.pi / 8.0, v, 0.0, 0.0]),
                                                     size=ObjectSize(1, 1, 1), confidence=1.0)
    dyn2 = DynamicObject.create_from_cartesian_state(obj_id=2, timestamp=35, cartesian_state=np.array([10.0, 0.0, np.pi / 8.0, v, 0.0, 0.0]),
                                                     size=ObjectSize(1, 1, 1), confidence=1.0)
    dynamic_objects = [dyn1, dyn2]
    size = ObjectSize(EGO_LENGTH, EGO_WIDTH, EGO_HEIGHT)
    ego_state = EgoState.create_from_cartesian_state(obj_id=0, timestamp=0, cartesian_state=np.array([1, 0, 0, 1.0, 0.0, 0]),
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
    ref_points = np.array([[x, -2.0] for x in range(0, 16)])
    ref_route = FrenetSerret2DFrame.fit(ref_points)
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
