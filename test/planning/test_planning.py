from unittest.mock import MagicMock, patch

from common_data.lcm.config import pubsub_topics
from common_data.src.communication.pubsub.pubsub import PubSub
from decision_making.src.planning.behavioral.behavioral_facade import BehavioralFacade
from decision_making.src.planning.behavioral.policies.semantic_actions_grid_policy import SemanticActionsGridPolicy
from decision_making.src.planning.behavioral.action_space.action_space import ActionSpaceContainer
from decision_making.src.planning.behavioral.action_space.dynamic_action_space import DynamicActionSpace
from decision_making.src.planning.behavioral.action_space.static_action_space import StaticActionSpace
from decision_making.src.planning.behavioral.behavioral_planning_facade import BehavioralPlanningFacade
from decision_making.src.planning.behavioral.evaluators.rule_based_action_spec_evaluator import \
    RuleBasedActionSpecEvaluator
from decision_making.src.planning.behavioral.evaluators.zero_value_approximator import ZeroValueApproximator
from decision_making.src.planning.behavioral.planner.single_step_behavioral_planner import SingleStepBehavioralPlanner
from decision_making.src.planning.trajectory.optimal_control.werling_planner import WerlingPlanner
from decision_making.src.planning.trajectory.trajectory_planning_facade import TrajectoryPlanningFacade
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.planning.trajectory.werling_planner import WerlingPlanner
from decision_making.src.prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.test.constants import MAP_SERVICE_ABSOLUTE_PATH
from mapping.test.model.testable_map_fixtures import map_api_mock

from decision_making.test.planning.custom_fixtures import pubsub, behavioral_facade, state_module, \
    navigation_facade, state, trajectory_params, behavioral_visualization_msg, navigation_plan


@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
def test_trajectoryPlanningFacade_realWerlingPlannerWithMocks_anyResult(pubsub: PubSub,
                                                                        behavioral_facade, state_module):
    # Using logger-mock here because facades catch exceptions and redirect them to logger
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
    action_space = ActionSpaceContainer(bp_logger,
                                        [StaticActionSpace(bp_logger), DynamicActionSpace(bp_logger, predictor)])
    planner = SingleStepBehavioralPlanner(action_space=action_space,
                                          recipe_evaluator=None,
                                          action_spec_evaluator=RuleBasedActionSpecEvaluator(bp_logger),
                                          action_spec_validator=None,
                                          value_approximator=ZeroValueApproximator(bp_logger),
                                          predictor=predictor, logger=bp_logger)

    state_module.periodic_action()
    navigation_facade.periodic_action()
    behavioral_planner_module = BehavioralPlanningFacade(pubsub=pubsub, logger=bp_logger, behavioral_planner=planner,
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

    # if this fails, that means BP did not publish a message - debug exceptions in BehavioralFacade
    behavioral_publish_mock.assert_called_once()
