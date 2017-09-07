from unittest.mock import patch, Mock, MagicMock

import numpy as np
import pytest

import decision_making.src.manager.dm_trigger
from decision_making.src.global_constants import *
from decision_making.src.manager.dm_process import DmProcess
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.messages.trajectory_parameters import SigmoidFunctionParams, TrajectoryCostParams, \
    TrajectoryParams
from decision_making.src.messages.trajectory_plan_message import TrajectoryPlanMsg
from decision_making.src.messages.visualization.behavioral_visualization_message import BehavioralVisualizationMsg
from decision_making.src.messages.visualization.trajectory_visualization_message import TrajectoryVisualizationMsg
from decision_making.src.planning.trajectory.optimal_control.werling_planner import WerlingPlanner
from decision_making.src.planning.trajectory.trajectory_planning_facade import TrajectoryPlanningFacade
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.state.state import OccupancyState, RoadLocalization, ObjectSize, EgoState, State
from decision_making.test.dds.mock_ddspubsub import DdsPubSubMock
from decision_making.test.planning.behavioral.mock_behavioral_facade import BehavioralFacadeMock
from decision_making.test.planning.navigation.mock_navigation_facade import NavigationFacadeMock
from decision_making.test.planning.trajectory.mock_trajectory_planning_facade import \
    TrajectoryPlanningFacadeMock
from decision_making.test.state.mock_state_module import StateModuleMock
from rte.python.logger.AV_logger import AV_Logger

#PATCH_DDS_PUBLISH = 'decision_making.test.dds.mock_ddspubsub.publish'

@pytest.fixture(scope='function')
def dds_pubsub():
    topic_pipeline = {
        NAVIGATION_PLAN_PUBLISH_TOPIC: [BEHAVIORAL_NAV_PLAN_READER_TOPIC],
        STATE_PUBLISH_TOPIC: [TRAJECTORY_STATE_READER_TOPIC, BEHAVIORAL_STATE_READER_TOPIC],
        BEHAVIORAL_TRAJECTORY_PARAMS_PUBLISH_TOPIC: [TRAJECTORY_PARAMS_READER_TOPIC]

    }
    dds = DdsPubSubMock(topic_pipeline)
    yield dds


@pytest.fixture(scope='function')
def state():
    occupancy_state = OccupancyState(0, np.array([]), np.array([]))
    dynamic_objects = []
    size = ObjectSize(0, 0, 0)
    road_localization = RoadLocalization(0, 0, 0, 0, 0, 0)
    ego_state = EgoState(0, 0, 0, 0, 0, 0, size, 0, 1.0, 0, 0, 0, 0, road_localization)
    yield State(occupancy_state, dynamic_objects, ego_state)


@pytest.fixture(scope='function')
def navigation_plan():
    yield NavigationPlanMsg(np.array([1, 2]))


@pytest.fixture(scope='function')
def trajectory_params():
    ref_route = np.array(
        [[1.0, -2.0], [2.0, -2.0], [3.0, -2.0], [4.0, -2.0], [5.0, -2.0], [6.0, -2.0],
         [7.0, -2.0], [8.0, -2.0], [9.0, -2.0], [10.0, -2.0], [11.0, -2.0],
         [12.0, -2.0], [13.0, -2.0], [14.0, -2.0], [15.0, -2.0], [16.0, -2.0]])
    target_state = np.array([16.0, -2.0, 0.0, 1])
    mock_sigmoid = SigmoidFunctionParams(1.0, 2.0, 3.0)
    trajectory_cost_params = TrajectoryCostParams(mock_sigmoid, mock_sigmoid, mock_sigmoid, mock_sigmoid,
                                                  mock_sigmoid, mock_sigmoid, mock_sigmoid, 16.0,
                                                  np.array([0.0, 2.0]), np.array([-1.0, 2.0]))
    yield TrajectoryParams(reference_route=ref_route, target_state=target_state,
                           cost_params=trajectory_cost_params, time=16,
                           strategy=TrajectoryPlanningStrategy.HIGHWAY)


@pytest.fixture(scope='function')
def trajectory():
    chosen_trajectory = np.array(
        [[1.0, 0.0, 0.0, 0.0], [2.0, -0.33, 0.0, 0.0], [3.0, -0.66, 0.0, 0.0], [4.0, -1.0, 0.0, 0.0],
         [5.0, -1.33, 0.0, 0.0], [6.0, -1.66, 0.0, 0.0], [7.0, -2.0, 0.0, 0.0], [8.0, -2.0, 0.0, 0.0],
         [9.0, -2.0, 0.0, 0.0], [10.0, -2.0, 0.0, 0.0], [11.0, -2.0, 0.0, 0.0]])
    ref_route = np.array(
        [[1.0, -2.0, 0.0], [2.0, -2.0, 0.0], [3.0, -2.0, 0.0], [4.0, -2.0, 0.0], [5.0, -2.0, 0.0],
         [6.0, -2.0, 0.0],
         [7.0, -2.0, 0.0], [8.0, -2.0, 0.0], [9.0, -2.0, 0.0], [10.0, -2.0, 0.0], [11.0, -2.0, 0.0],
         [12.0, -2.0, 0.0], [13.0, -2.0, 0.0], [14.0, -2.0, 0.0], [15.0, -2.0, 0.0], [16.0, -2.0, 0.0]])
    yield TrajectoryPlanMsg(trajectory=chosen_trajectory, reference_route=ref_route, current_speed=5.0)


@pytest.fixture(scope='function')
def behavioral_visualization_msg(trajectory_params):
    yield BehavioralVisualizationMsg(trajectory_params.reference_route)


@pytest.fixture(scope='function')
def trajectory_visualization_msg(trajectory):
    yield TrajectoryVisualizationMsg(reference_route=trajectory.reference_route,
                                     trajectories=np.array([trajectory.chosen_trajectory])
                                     , costs=np.array([0]))


@pytest.fixture(scope='function')
def state_module(dds_pubsub, state):
    # dds_pubsub = DdsPubSub(STATE_MODULE_DDS_PARTICIPANT, DECISION_MAKING_DDS_FILE)

    logger = AV_Logger.get_logger(STATE_MODULE_NAME_FOR_LOGGING)

    state_mock = StateModuleMock(dds_pubsub, logger, state)
    state_mock.start()
    yield state_mock
    state_mock.stop()


@pytest.fixture(scope='function')
def behavioral_facade(dds_pubsub, trajectory_params, behavioral_visualization_msg):
    logger = AV_Logger.get_logger(BEHAVIORAL_PLANNING_NAME_FOR_LOGGING)

    behavioral_module = BehavioralFacadeMock(dds=dds_pubsub, logger=logger, trajectory_params=trajectory_params,
                                             visualization_msg=behavioral_visualization_msg)

    behavioral_module.start()
    yield behavioral_module
    behavioral_module.stop()


@pytest.fixture(scope='function')
def navigation_facade(dds_pubsub, navigation_plan):
    logger = AV_Logger.get_logger(NAVIGATION_PLANNING_NAME_FOR_LOGGING)
    # dds = DdsPubSub(NAVIGATION_PLANNER_DDS_PARTICIPANT, DECISION_MAKING_DDS_FILE)

    navigation_module = NavigationFacadeMock(dds=dds_pubsub, logger=logger, navigation_plan_msg=navigation_plan)

    navigation_module.start()
    yield navigation_module
    navigation_module.stop()


@pytest.fixture(scope='function')
def trajectory_planner_facade(dds_pubsub, trajectory, trajectory_visualization_msg):
    logger = AV_Logger.get_logger(TRAJECTORY_PLANNING_NAME_FOR_LOGGING)
    # dds_pubsub = DdsPubSub(TRAJECTORY_PLANNER_DDS_PARTICIPANT, DECISION_MAKING_DDS_FILE)

    trajectory_planning_module = TrajectoryPlanningFacadeMock(dds=dds_pubsub, logger=logger,
                                                              trajectory_msg=trajectory,
                                                              visualization_msg=trajectory_visualization_msg)
    trajectory_planning_module.start()
    yield trajectory_planning_module
    trajectory_planning_module.stop()

def test_trajectory(dds_pubsub: DdsPubSubMock, state_module, behavioral_facade):
    logger = AV_Logger.get_logger(TRAJECTORY_PLANNING_NAME_FOR_LOGGING)

    planner = WerlingPlanner(logger)
    strategy_handlers = {TrajectoryPlanningStrategy.HIGHWAY: planner,
                         TrajectoryPlanningStrategy.PARKING: planner,
                         TrajectoryPlanningStrategy.TRAFFIC_JAM: planner}

    trajectory_planning_module = TrajectoryPlanningFacade(dds=dds_pubsub, logger=logger,
                                                          strategy_handlers=strategy_handlers)


    m = MagicMock()
    dds_pubsub.subscribe(TRAJECTORY_PUBLISH_TOPIC, m)

    trajectory_planning_module.start()
    behavioral_facade.periodic_action()
    state_module.periodic_action()
    trajectory_planning_module.periodic_action()

    m.assert_called_once()





#
# @pytest.fixture(scope='function')
# def control_simulation_facade(request, dds_pubsub):
#     logger = AV_Logger.get_logger(CONTROL_PLANNER_NAME_FOR_LOGGING)
#
#     control_simulation_module = ControlSimulationFacade(dds=dds_pubsub, logger=logger,
#                                                         control_handler=SimpleController(),
#                                                         state_period=request.param[Input.STATE_PERIOD],
#                                                         trajectory_period=request.param[Input.TRAJECTORY_PERIOD])
#
#     control_simulation_module.start()
#     yield control_simulation_module
#     control_simulation_module.stop()
