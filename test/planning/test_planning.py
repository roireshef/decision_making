from unittest.mock import MagicMock, patch

from common_data.src.communication.pubsub.pubsub import PubSub
from decision_making.src.global_constants import TRAJECTORY_PLANNING_NAME_FOR_LOGGING, \
    BEHAVIORAL_PLANNING_NAME_FOR_LOGGING
from decision_making.src.planning.behavioral.behavioral_facade import BehavioralFacade
from decision_making.src.planning.behavioral.policies.semantic_actions_grid_policy import SemanticActionsGridPolicy
from decision_making.src.planning.trajectory.optimal_control.werling_planner import WerlingPlanner
from decision_making.src.planning.trajectory.trajectory_planning_facade import TrajectoryPlanningFacade
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.prediction.road_following_predictor import RoadFollowingPredictor
from common_data.lcm.config import pubsub_topics
from decision_making.test.constants import MAP_SERVICE_ABSOLUTE_PATH
from decision_making.test.planning.custom_fixtures import pubsub, behavioral_facade, state_module, \
    navigation_facade, state, trajectory_params, behavioral_visualization_msg, navigation_plan

from mapping.test.model.testable_map_fixtures import map_api_mock

from rte.python.logger.AV_logger import AV_Logger

# TODO: need to add a logger-mock here because facades catch exceptions and redirect them to logger
@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
def test_trajectoryPlanningFacade_realWerlingPlannerWithMocks_anyResult(pubsub: PubSub,
                                                                        behavioral_facade, state_module):
    tp_logger = MagicMock()
    predictor_logger = MagicMock()

    trajectory_publish_mock = MagicMock()
    predictor = RoadFollowingPredictor(predictor_logger)

    planner = WerlingPlanner(tp_logger, predictor)
    strategy_handlers = {TrajectoryPlanningStrategy.HIGHWAY: planner,
                         TrajectoryPlanningStrategy.PARKING: planner,
                         TrajectoryPlanningStrategy.TRAFFIC_JAM: planner}

    trajectory_planning_module = TrajectoryPlanningFacade(pubsub=pubsub, logger=tp_logger,
                                                          strategy_handlers=strategy_handlers,
                                                          short_time_predictor=predictor)

    pubsub.subscribe(pubsub_topics.TRAJECTORY_TOPIC, trajectory_publish_mock)

    state_module.periodic_action()
    trajectory_planning_module.start()
    behavioral_facade.periodic_action()
    state_module.periodic_action()
    trajectory_planning_module.periodic_action()

    tp_logger.warn.assert_not_called()
    tp_logger.error.assert_not_called()
    tp_logger.critical.assert_not_called()

    predictor_logger.warn.assert_not_called()
    predictor_logger.error.assert_not_called()
    predictor_logger.critical.assert_not_called()

    trajectory_publish_mock.assert_called_once()


@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
def test_behavioralPlanningFacade_semanticPolicy_anyResult(pubsub: PubSub, state_module,
                                                           navigation_facade):
    bp_logger = MagicMock()
    predictor_logger = MagicMock()

    behavioral_publish_mock = MagicMock()
    predictor = RoadFollowingPredictor(predictor_logger)
    policy = SemanticActionsGridPolicy(bp_logger, predictor)

    state_module.periodic_action()
    navigation_facade.periodic_action()
    behavioral_planner_module = BehavioralFacade(pubsub=pubsub, logger=bp_logger, policy=policy,
                                                 short_time_predictor=predictor)

    pubsub.subscribe(pubsub_topics.TRAJECTORY_PARAMS_TOPIC, behavioral_publish_mock)

    bp_logger.warn.assert_not_called()
    bp_logger.error.assert_not_called()
    bp_logger.critical.assert_not_called()

    predictor_logger.warn.assert_not_called()
    predictor_logger.error.assert_not_called()
    predictor_logger.critical.assert_not_called()


    behavioral_planner_module.start()
    behavioral_planner_module.periodic_action()

    behavioral_publish_mock.assert_called_once()
