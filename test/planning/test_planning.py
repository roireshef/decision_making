from decision_making.test.messages.scene_static_fixture import scene_static_short_testable
from unittest.mock import MagicMock

from decision_making.src.infra.pubsub import PubSub
from common_data.interface.Rte_Types.python.uc_system import UC_SYSTEM_TRAJECTORY_PLAN
from common_data.interface.Rte_Types.python.uc_system import UC_SYSTEM_SCENE_STATIC
from common_data.interface.Rte_Types.python.uc_system import UC_SYSTEM_SCENE_DYNAMIC
from common_data.interface.Rte_Types.python.uc_system import UC_SYSTEM_TRAJECTORY_PARAMS
from decision_making.src.planning.route.route_planning_facade import RoutePlanningFacade

from decision_making.src.scene.scene_static_model import SceneStaticModel
from decision_making.src.planning.behavioral.action_space.action_space import ActionSpaceContainer
from decision_making.src.planning.behavioral.action_space.dynamic_action_space import DynamicActionSpace
from decision_making.src.planning.behavioral.action_space.static_action_space import StaticActionSpace
from decision_making.src.planning.behavioral.behavioral_planning_facade import BehavioralPlanningFacade
from decision_making.src.planning.behavioral.evaluators.single_lane_action_spec_evaluator import SingleLaneActionSpecEvaluator
from decision_making.src.planning.behavioral.evaluators.zero_value_approximator import ZeroValueApproximator
from decision_making.src.planning.behavioral.filtering.action_spec_filter_bank import FilterIfNone
from decision_making.src.planning.behavioral.filtering.action_spec_filtering import ActionSpecFiltering
from decision_making.src.planning.behavioral.planner.single_step_behavioral_planner import SingleStepBehavioralPlanner
from decision_making.src.planning.trajectory.trajectory_planning_facade import TrajectoryPlanningFacade
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.planning.trajectory.werling_planner import WerlingPlanner
from decision_making.src.prediction.ego_aware_prediction.road_following_predictor import RoadFollowingPredictor

from decision_making.src.planning.behavioral.default_config import DEFAULT_DYNAMIC_RECIPE_FILTERING, \
    DEFAULT_STATIC_RECIPE_FILTERING

from decision_making.test.planning.custom_fixtures import pubsub, behavioral_facade, \
    state, trajectory_params, behavioral_visualization_msg, route_planner_facade, route_plan_1_2, scene_dynamic
from decision_making.test.messages.scene_static_fixture import scene_static_short_testable


def test_trajectoryPlanningFacade_realWerlingPlannerWithMocks_anyResult(pubsub: PubSub,
                                                                        behavioral_facade: BehavioralPlanningFacade,
                                                                        scene_static_short_testable,
                                                                        scene_dynamic):
    SceneStaticModel.get_instance().set_scene_static(scene_static_short_testable)

    # Using logger-mock here because facades catch exceptions and redirect them to logger
    tp_logger = MagicMock()
    predictor_logger = MagicMock()

    trajectory_publish_mock = MagicMock()
    predictor = RoadFollowingPredictor(predictor_logger)

    planner = WerlingPlanner(tp_logger, predictor)
    strategy_handlers = {TrajectoryPlanningStrategy.HIGHWAY: planner,
                         TrajectoryPlanningStrategy.PARKING: planner,
                         TrajectoryPlanningStrategy.TRAFFIC_JAM: planner}

    trajectory_facade = TrajectoryPlanningFacade(pubsub=pubsub, logger=tp_logger,
                                                 strategy_handlers=strategy_handlers)

    pubsub.subscribe(UC_SYSTEM_TRAJECTORY_PLAN, trajectory_publish_mock)

    pubsub.publish(UC_SYSTEM_SCENE_DYNAMIC, scene_dynamic.serialize())
    trajectory_facade.start()

    pubsub.publish(UC_SYSTEM_SCENE_STATIC, scene_static_short_testable.serialize())

    behavioral_facade.periodic_action()
    pubsub.publish(UC_SYSTEM_SCENE_DYNAMIC, scene_dynamic.serialize())
    trajectory_facade.periodic_action()

    # if this fails, that means BP did not publish a message - debug exceptions in TrajectoryPlanningFacade
    tp_logger.warn.assert_not_called()
    tp_logger.error.assert_not_called()
    tp_logger.critical.assert_not_called()

    predictor_logger.warn.assert_not_called()
    predictor_logger.error.assert_not_called()
    predictor_logger.critical.assert_not_called()

    trajectory_publish_mock.assert_called_once()

    pubsub.unsubscribe(UC_SYSTEM_TRAJECTORY_PLAN)


def test_behavioralPlanningFacade_arbitraryState_returnsAnyResult(pubsub: PubSub,
                                                                  route_planner_facade: RoutePlanningFacade,
                                                                  scene_static_short_testable,
                                                                  scene_dynamic):

    SceneStaticModel.get_instance().set_scene_static(scene_static_short_testable)

    bp_logger = MagicMock()
    predictor_logger = MagicMock()

    behavioral_publish_mock = MagicMock()
    predictor = RoadFollowingPredictor(predictor_logger)
    action_space = ActionSpaceContainer(bp_logger,
                                        [StaticActionSpace(bp_logger, filtering=DEFAULT_STATIC_RECIPE_FILTERING),
                                         DynamicActionSpace(bp_logger, predictor,
                                                            filtering=DEFAULT_DYNAMIC_RECIPE_FILTERING)])
    planner = SingleStepBehavioralPlanner(action_space=action_space,
                                          recipe_evaluator=None,
                                          action_spec_evaluator=SingleLaneActionSpecEvaluator(bp_logger),
                                          action_spec_validator=ActionSpecFiltering(filters=[FilterIfNone()],
                                                                                    logger=bp_logger),
                                          value_approximator=ZeroValueApproximator(bp_logger),
                                          predictor=predictor, logger=bp_logger)

    pubsub.publish(UC_SYSTEM_SCENE_DYNAMIC, scene_dynamic.serialize())
    route_planner_facade.periodic_action()

    behavioral_planner_module = BehavioralPlanningFacade(pubsub=pubsub, logger=bp_logger, behavioral_planner=planner)

    pubsub.subscribe(UC_SYSTEM_TRAJECTORY_PARAMS, behavioral_publish_mock)

    bp_logger.warn.assert_not_called()
    bp_logger.error.assert_not_called()
    bp_logger.critical.assert_not_called()

    predictor_logger.warn.assert_not_called()
    predictor_logger.error.assert_not_called()
    predictor_logger.critical.assert_not_called()

    pubsub.publish(UC_SYSTEM_SCENE_STATIC, scene_static_short_testable.serialize())
    behavioral_planner_module.start()
    behavioral_planner_module.periodic_action()

    # if this fails, that means BP did not publish a message - debug exceptions in BehavioralFacade
    behavioral_publish_mock.assert_called_once()

    pubsub.unsubscribe(UC_SYSTEM_TRAJECTORY_PARAMS)
