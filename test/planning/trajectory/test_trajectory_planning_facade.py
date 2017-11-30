from unittest.mock import patch, MagicMock

import pytest

from decision_making.src.global_constants import NEGLIGIBLE_DISPOSITION_LAT, TRAJECTORY_PUBLISH_TOPIC
from decision_making.src.planning.trajectory.optimal_control.werling_planner import WerlingPlanner
from decision_making.src.planning.trajectory.trajectory_planning_facade import TrajectoryPlanningFacade
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.state.state import OccupancyState, EgoState, ObjectSize, State
from decision_making.test.constants import MAP_SERVICE_ABSOLUTE_PATH
from decision_making.test.dds.mock_ddspubsub import DdsPubSubMock
from decision_making.test.state.mock_state_module import StateModuleMock
from mapping.test.model.testable_map_fixtures import map_api_mock, testable_map_api
from rte.python.logger.AV_logger import AV_Logger
from decision_making.test.planning.custom_fixtures import dds_pubsub, behavioral_facade, behavioral_visualization_msg,\
    trajectory_params

import numpy as np

EXPECTED_EGO_PATH = "decision_making.src.planning.trajectory.trajectory_planning_facade." \
                    "TrajectoryPlanningFacade._get_state_with_expected_ego"

@pytest.fixture(scope='function')
def empty_state_ego_at_initial_point():
    occupancy_state = OccupancyState(0, np.array([]), np.array([]))
    dynamic_objects = []
    ego_state = EgoState(0, 0, 0.0, -2.0, 0, 0, ObjectSize(0, 0, 0), 0, 1.0, 0, 0, 0, 0)
    yield State(occupancy_state, dynamic_objects, ego_state)


@pytest.fixture(scope='function')
def empty_state_ego_at_1sec_little_disturbance():
    occupancy_state = OccupancyState(0, np.array([]), np.array([]))
    dynamic_objects = []
    ego_state = EgoState(0, int(1e9), 1.0, -2 + NEGLIGIBLE_DISPOSITION_LAT/2, 0, 0, ObjectSize(0, 0, 0), 0, 1.0, 0, 0, 0, 0)
    yield State(occupancy_state, dynamic_objects, ego_state)


@pytest.fixture(scope='function')
def empty_state_ego_at_1sec_big_disturbance():
    occupancy_state = OccupancyState(0, np.array([]), np.array([]))
    dynamic_objects = []
    ego_state = EgoState(0, int(1e9), 1.0, -2 + NEGLIGIBLE_DISPOSITION_LAT * 2, 0, 0, ObjectSize(0, 0, 0), 0, 1.0, 0, 0, 0, 0)
    yield State(occupancy_state, dynamic_objects, ego_state)


# TODO: need to add a logger-mock here because facades catch exceptions and redirect them to logger
@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
@patch(target=EXPECTED_EGO_PATH)
def test_trajectoryPlanningFacade_twoPlanningIterationsLittleDistrubance_replanningTakesExpectedEgoState(
        get_expected_state_magic_mock: MagicMock, dds_pubsub: DdsPubSubMock, behavioral_facade,
        empty_state_ego_at_initial_point, empty_state_ego_at_1sec_little_disturbance):

    logger = AV_Logger.get_logger("test_trajectoryPlanningFacade_logger")

    predictor = RoadFollowingPredictor(logger)
    planner = WerlingPlanner(logger, predictor)

    strategy_handlers = {TrajectoryPlanningStrategy.HIGHWAY: planner,
                         TrajectoryPlanningStrategy.PARKING: planner,
                         TrajectoryPlanningStrategy.TRAFFIC_JAM: planner}

    trajectory_planning_facade = TrajectoryPlanningFacade(dds=dds_pubsub, logger=logger,
                                                          strategy_handlers=strategy_handlers)

    initial_state_module_mock = StateModuleMock(dds_pubsub, logger, empty_state_ego_at_initial_point)
    secondary_state_module_mock = StateModuleMock(dds_pubsub, logger, empty_state_ego_at_1sec_little_disturbance)

    trajectory_planning_facade.start()
    behavioral_facade.periodic_action()

    # send first state and call TP facade once
    initial_state_module_mock.start()
    initial_state_module_mock.periodic_action()
    trajectory_planning_facade.periodic_action()
    initial_state_module_mock.stop()

    # redirect second message from TP to our mock
    trajectory_publish_mock = MagicMock()
    dds_pubsub.subscribe(TRAJECTORY_PUBLISH_TOPIC, trajectory_publish_mock)

    # send second state and call TP facade again
    secondary_state_module_mock.start()
    secondary_state_module_mock.periodic_action()
    trajectory_planning_facade.periodic_action()
    secondary_state_module_mock.stop()

    # assert TP used TrajectoryPlanningFacade._get_state_with_expected_ego
    # (since it is mocked and overridden, TrajectoryPlanningFacade will fail to complete the second execution).
    get_expected_state_magic_mock.assert_called_once()

@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
@patch(target=EXPECTED_EGO_PATH)
def test_trajectoryPlanningFacade_twoPlanningIterationsBigDistrubance_replanningTakesActualEgoState(
        get_expected_state_magic_mock: MagicMock, dds_pubsub: DdsPubSubMock, behavioral_facade,
        empty_state_ego_at_initial_point, empty_state_ego_at_1sec_big_disturbance):

    logger = AV_Logger.get_logger("test_trajectoryPlanningFacade_logger")

    predictor = RoadFollowingPredictor(logger)
    planner = WerlingPlanner(logger, predictor)

    strategy_handlers = {TrajectoryPlanningStrategy.HIGHWAY: planner,
                         TrajectoryPlanningStrategy.PARKING: planner,
                         TrajectoryPlanningStrategy.TRAFFIC_JAM: planner}

    trajectory_planning_facade = TrajectoryPlanningFacade(dds=dds_pubsub, logger=logger,
                                                          strategy_handlers=strategy_handlers)

    initial_state_module_mock = StateModuleMock(dds_pubsub, logger, empty_state_ego_at_initial_point)
    secondary_state_module_mock = StateModuleMock(dds_pubsub, logger, empty_state_ego_at_1sec_big_disturbance)

    trajectory_planning_facade.start()
    behavioral_facade.periodic_action()

    # send first state and call TP facade once
    initial_state_module_mock.start()
    initial_state_module_mock.periodic_action()
    trajectory_planning_facade.periodic_action()
    initial_state_module_mock.stop()

    # redirect second message from TP to our mock
    trajectory_publish_mock = MagicMock()
    dds_pubsub.subscribe(TRAJECTORY_PUBLISH_TOPIC, trajectory_publish_mock)

    # send second state and call TP facade again
    secondary_state_module_mock.start()
    secondary_state_module_mock.periodic_action()
    trajectory_planning_facade.periodic_action()
    secondary_state_module_mock.stop()

    # assert TP used TrajectoryPlanningFacade._get_state_with_expected_ego
    get_expected_state_magic_mock.assert_not_called()
    trajectory_publish_mock.assert_called_once()
