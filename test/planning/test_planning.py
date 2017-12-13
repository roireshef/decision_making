from unittest.mock import MagicMock, patch

from decision_making.src.global_constants import TRAJECTORY_PLANNING_NAME_FOR_LOGGING, TRAJECTORY_PUBLISH_TOPIC, \
    BEHAVIORAL_PLANNING_NAME_FOR_LOGGING, TRAJECTORY_PARAMS_READER_TOPIC
from decision_making.src.planning.behavioral.behavioral_facade import BehavioralFacade
from decision_making.src.planning.behavioral.policies.semantic_actions_grid_policy import SemanticActionsGridPolicy
from decision_making.src.planning.behavioral.semantic_actions_policy import SemanticActionsPolicy
from decision_making.src.planning.trajectory.optimal_control.werling_planner import WerlingPlanner
from decision_making.src.planning.trajectory.trajectory_planning_facade import TrajectoryPlanningFacade
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.prediction.predictor import Predictor
from decision_making.src.prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.test.constants import MAP_SERVICE_ABSOLUTE_PATH
from decision_making.test.dds.mock_ddspubsub import DdsPubSubMock
from decision_making.test.planning.custom_fixtures import *

from mapping.test.model.testable_map_fixtures import map_api_mock, testable_map_api

from rte.python.logger.AV_logger import AV_Logger


@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
def test_trajectoryPlanningFacade_realWerlingPlannerWithMocks_anyResult(dds_pubsub: DdsPubSubMock,
                                                                        behavioral_facade, state_module):
    logger = AV_Logger.get_logger(TRAJECTORY_PLANNING_NAME_FOR_LOGGING)
    trajectory_publish_mock = MagicMock()
    predictor = RoadFollowingPredictor(logger)

    planner = WerlingPlanner(logger, predictor)
    strategy_handlers = {TrajectoryPlanningStrategy.HIGHWAY: planner,
                         TrajectoryPlanningStrategy.PARKING: planner,
                         TrajectoryPlanningStrategy.TRAFFIC_JAM: planner}

    trajectory_planning_module = TrajectoryPlanningFacade(dds=dds_pubsub, logger=logger,
                                                          strategy_handlers=strategy_handlers)

    dds_pubsub.subscribe(TRAJECTORY_PUBLISH_TOPIC, trajectory_publish_mock)

    trajectory_planning_module.start()
    behavioral_facade.periodic_action()
    state_module.periodic_action()
    trajectory_planning_module.periodic_action()

    trajectory_publish_mock.assert_called_once()


@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
def test_behavioralPlanningFacade_semanticPolicy_anyResult(dds_pubsub: DdsPubSubMock, state_module,
                                                           navigation_facade):
    logger = AV_Logger.get_logger(BEHAVIORAL_PLANNING_NAME_FOR_LOGGING)
    behavioral_publish_mock = MagicMock()
    predictor = RoadFollowingPredictor(logger)
    policy = SemanticActionsGridPolicy(logger, predictor)

    state_module.periodic_action()
    navigation_facade.periodic_action()
    behavioral_planner_module = BehavioralFacade(dds=dds_pubsub, logger=logger, policy=policy)

    dds_pubsub.subscribe(TRAJECTORY_PARAMS_READER_TOPIC, behavioral_publish_mock)
    # dds_pubsub.subscribe(STATE_PUBLISH_TOPIC, state_module)

    behavioral_planner_module.start()
    behavioral_planner_module.periodic_action()

    # behavioral_publish_mock.periodic_action()

    behavioral_publish_mock.assert_called_once()
