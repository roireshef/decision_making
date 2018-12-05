from unittest.mock import MagicMock, patch

from common_data.interface.py.pubsub import Rte_Types_pubsub_topics as pubsub_topics
from common_data.src.communication.pubsub.pubsub import PubSub
from decision_making.src.mapping.scene_model import SceneModel
from decision_making.src.messages.scene_static_message import SceneStatic
from decision_making.src.planning.behavioral.action_space.action_space import ActionSpaceContainer
from decision_making.src.planning.behavioral.action_space.dynamic_action_space import DynamicActionSpace
from decision_making.src.planning.behavioral.action_space.static_action_space import StaticActionSpace
from decision_making.src.planning.behavioral.behavioral_planning_facade import BehavioralPlanningFacade
from decision_making.src.planning.behavioral.evaluators.rule_based_action_spec_evaluator import \
    RuleBasedActionSpecEvaluator
from decision_making.src.planning.behavioral.evaluators.zero_value_approximator import ZeroValueApproximator
from decision_making.src.planning.behavioral.filtering.action_spec_filter_bank import FilterIfNone
from decision_making.src.planning.behavioral.filtering.action_spec_filtering import ActionSpecFiltering
from decision_making.src.planning.behavioral.planner.single_step_behavioral_planner import SingleStepBehavioralPlanner
from decision_making.src.planning.navigation.navigation_facade import NavigationFacade
from decision_making.src.planning.trajectory.trajectory_planning_facade import TrajectoryPlanningFacade
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.planning.trajectory.werling_planner import WerlingPlanner
from decision_making.src.prediction.ego_aware_prediction.road_following_predictor import RoadFollowingPredictor

from decision_making.src.prediction.action_unaware_prediction.physical_time_alignment_predictor import \
    PhysicalTimeAlignmentPredictor
from decision_making.src.state.state_module import StateModule
from decision_making.test.constants import MAP_SERVICE_ABSOLUTE_PATH
from mapping.test.model.testable_map_fixtures import map_api_mock

from decision_making.src.planning.behavioral.default_config import DEFAULT_DYNAMIC_RECIPE_FILTERING, \
    DEFAULT_STATIC_RECIPE_FILTERING

from decision_making.test.planning.custom_fixtures import pubsub, behavioral_facade, state_module, \
    navigation_facade, state, trajectory_params, behavioral_visualization_msg, navigation_plan

from decision_making.test.messages.static_scene_fixture import scene_static_no_split, scene_static

@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
def test_trajectoryPlanningFacade_realWerlingPlannerWithMocks_anyResult(pubsub: PubSub,
                                                                        behavioral_facade: BehavioralPlanningFacade,
                                                                        state_module:StateModule,
                                                                        scene_static: SceneStatic,
                                                                        scene_static_no_split: SceneStatic):

    SceneModel.get_instance().set_scene_static(scene_static_no_split)
    # Using logger-mock here because facades catch exceptions and redirect them to logger
    tp_logger = MagicMock()
    predictor_logger = MagicMock()

    trajectory_publish_mock = MagicMock()
    predictor = RoadFollowingPredictor(predictor_logger)
    short_time_predictor = PhysicalTimeAlignmentPredictor(predictor_logger)

    planner = WerlingPlanner(tp_logger, predictor)
    strategy_handlers = {TrajectoryPlanningStrategy.HIGHWAY: planner,
                         TrajectoryPlanningStrategy.PARKING: planner,
                         TrajectoryPlanningStrategy.TRAFFIC_JAM: planner}

    trajectory_facade = TrajectoryPlanningFacade(pubsub=pubsub, logger=tp_logger,
                                                 strategy_handlers=strategy_handlers,
                                                 short_time_predictor=short_time_predictor)

    pubsub.subscribe(pubsub_topics.TRAJECTORY_PLAN, trajectory_publish_mock)

    state_module.periodic_action()
    trajectory_facade.start()

    pubsub.publish(pubsub_topics.SCENE_STATIC, scene_static.serialize())

    behavioral_facade.periodic_action()
    state_module.periodic_action()
    trajectory_facade.periodic_action()

    # if this fails, that means BP did not publish a message - debug exceptions in TrajectoryPlanningFacade
    tp_logger.warn.assert_not_called()
    tp_logger.error.assert_not_called()
    tp_logger.critical.assert_not_called()

    predictor_logger.warn.assert_not_called()
    predictor_logger.error.assert_not_called()
    predictor_logger.critical.assert_not_called()

    trajectory_publish_mock.assert_called_once()


@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
def test_behavioralPlanningFacade_arbitraryState_returnsAnyResult(pubsub: PubSub, state_module:StateModule,
                                                                  navigation_facade: NavigationFacade,
                                                                  scene_static: SceneStatic):

    SceneModel.get_instance().set_scene_static(scene_static)
    bp_logger = MagicMock()
    predictor_logger = MagicMock()

    behavioral_publish_mock = MagicMock()
    predictor = RoadFollowingPredictor(predictor_logger)
    short_time_predictor = PhysicalTimeAlignmentPredictor(predictor_logger)
    action_space = ActionSpaceContainer(bp_logger,
                                        [StaticActionSpace(bp_logger, filtering=DEFAULT_STATIC_RECIPE_FILTERING),
                                         DynamicActionSpace(bp_logger, predictor,
                                                            filtering=DEFAULT_DYNAMIC_RECIPE_FILTERING)])
    planner = SingleStepBehavioralPlanner(action_space=action_space,
                                          recipe_evaluator=None,
                                          action_spec_evaluator=RuleBasedActionSpecEvaluator(bp_logger),
                                          action_spec_validator=ActionSpecFiltering(filters=[FilterIfNone()],
                                                                                    logger=bp_logger),
                                          value_approximator=ZeroValueApproximator(bp_logger),
                                          predictor=predictor, logger=bp_logger)

    state_module.periodic_action()
    navigation_facade.periodic_action()
    behavioral_planner_module = BehavioralPlanningFacade(pubsub=pubsub, logger=bp_logger, behavioral_planner=planner,
                                                         short_time_predictor=short_time_predictor)

    pubsub.subscribe(pubsub_topics.TRAJECTORY_PARAMS_LCM, behavioral_publish_mock)

    bp_logger.warn.assert_not_called()
    bp_logger.error.assert_not_called()
    bp_logger.critical.assert_not_called()

    predictor_logger.warn.assert_not_called()
    predictor_logger.error.assert_not_called()
    predictor_logger.critical.assert_not_called()

    pubsub.publish(pubsub_topics.SCENE_STATIC, scene_static.serialize())
    behavioral_planner_module.start()
    behavioral_planner_module.periodic_action()


    # if this fails, that means BP did not publish a message - debug exceptions in BehavioralFacade
    behavioral_publish_mock.assert_called_once()
