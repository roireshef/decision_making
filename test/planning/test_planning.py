from unittest.mock import MagicMock, patch

from common_data.src.communication.pubsub.pubsub import PubSub
from decision_making.src.global_constants import TRAJECTORY_PLANNING_NAME_FOR_LOGGING, \
                                                 BEHAVIORAL_PLANNING_NAME_FOR_LOGGING
from decision_making.src.planning.behavioral.behavioral_facade import BehavioralFacade
from decision_making.src.planning.behavioral.policies.semantic_actions_grid_policy import SemanticActionsGridPolicy
from decision_making.src.planning.behavioral.semantic_actions_policy import SemanticActionsPolicy
from decision_making.src.planning.trajectory.optimal_control.werling_planner import WerlingPlanner
from decision_making.src.planning.trajectory.trajectory_planning_facade import TrajectoryPlanningFacade
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.prediction.predictor import Predictor
from decision_making.src.prediction.road_following_predictor import RoadFollowingPredictor
from common_data.lcm.config import pubsub_topics
from decision_making.test.constants import MAP_SERVICE_ABSOLUTE_PATH
from decision_making.test.planning.custom_fixtures import pubsub, behavioral_facade, state_module, navigation_facade
from common_data.lcm.generatedFiles.gm_lcm.LcmTrajectoryParameters import LcmTrajectoryParameters
from common_data.lcm.generatedFiles.gm_lcm.LcmTrajectoryData import LcmTrajectoryData

from mapping.test.model.testable_map_fixtures import map_api_mock

from rte.python.logger.AV_logger import AV_Logger


@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
def test_trajectoryPlanningFacade_realWerlingPlannerWithMocks_anyResult(pubsub: PubSub,
                                                                        behavioral_facade, state_module):
    logger = AV_Logger.get_logger(TRAJECTORY_PLANNING_NAME_FOR_LOGGING)
    trajectory_publish_mock = MagicMock()
    predictor = RoadFollowingPredictor(logger)

    planner = WerlingPlanner(logger, predictor)
    strategy_handlers = {TrajectoryPlanningStrategy.HIGHWAY: planner,
                         TrajectoryPlanningStrategy.PARKING: planner,
                         TrajectoryPlanningStrategy.TRAFFIC_JAM: planner}

    trajectory_planning_module = TrajectoryPlanningFacade(pubsub=pubsub, logger=logger,
                                                          strategy_handlers=strategy_handlers)

    pubsub.subscribe(pubsub_topics.TRAJECTORY_TOPIC, trajectory_publish_mock)

    trajectory_planning_module.start()
    behavioral_facade.periodic_action()
    state_module.periodic_action()
    trajectory_planning_module.periodic_action()

    trajectory_publish_mock.assert_called_once()


@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
def test_behavioralPlanningFacade_semanticPolicy_anyResult(pubsub: PubSub, state_module,
                                                           navigation_facade):
    logger = AV_Logger.get_logger(BEHAVIORAL_PLANNING_NAME_FOR_LOGGING)
    behavioral_publish_mock = MagicMock()
    predictor = RoadFollowingPredictor(logger)
    policy = SemanticActionsGridPolicy(logger, predictor)

    state_module.periodic_action()
    navigation_facade.periodic_action()
    behavioral_planner_module = BehavioralFacade(pubsub=pubsub, logger=logger, policy=policy)

    pubsub.subscribe(pubsub_topics.TRAJECTORY_PARAMS_TOPIC, behavioral_publish_mock)

    behavioral_planner_module.start()
    behavioral_planner_module.periodic_action()

    # behavioral_publish_mock.periodic_action()

    behavioral_publish_mock.assert_called_once()

