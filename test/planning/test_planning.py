from unittest.mock import MagicMock

from decision_making.src.global_constants import TRAJECTORY_PLANNING_NAME_FOR_LOGGING, TRAJECTORY_PUBLISH_TOPIC, \
    BEHAVIORAL_PLANNING_NAME_FOR_LOGGING, TRAJECTORY_PARAMS_READER_TOPIC
from decision_making.src.planning.behavioral.behavioral_facade import BehavioralFacade
from decision_making.src.planning.behavioral.default_policy import DefaultPolicy
from decision_making.src.planning.behavioral.default_policy_config import DefaultPolicyConfig
from decision_making.src.planning.trajectory.optimal_control.werling_planner import WerlingPlanner
from decision_making.src.planning.trajectory.trajectory_planning_facade import TrajectoryPlanningFacade
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.prediction.predictor import Predictor
from decision_making.src.prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.test.dds.mock_ddspubsub import DdsPubSubMock
from decision_making.test.planning.custom_fixtures import state_module, behavioral_facade, dds_pubsub
from mapping.test.model.testable_map_fixtures import testable_map_api

from rte.python.logger.AV_logger import AV_Logger


def test_trajectoryPlanningFacade_realWerlingPlannerWithMocks_anyResult(
        dds_pubsub: DdsPubSubMock, state_module, behavioral_facade, testable_map_api):
    logger = AV_Logger.get_logger(TRAJECTORY_PLANNING_NAME_FOR_LOGGING)
    trajectory_publish_mock = MagicMock()
    predictor = Predictor(testable_map_api)

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


def test_behavioralPlanningFacade_defaultPolicy_anyResult(dds_pubsub: DdsPubSubMock, state_module, navigation_facade,
                                                          default_policy_behavioral_state):
    logger = AV_Logger.get_logger(BEHAVIORAL_PLANNING_NAME_FOR_LOGGING)
    behavioral_publish_mock = MagicMock()
    policy_config = DefaultPolicyConfig()
    policy = DefaultPolicy(logger, policy_config, default_policy_behavioral_state, None, None)

    state_module.periodic_action()
    navigation_facade.periodic_action()
    behavioral_planner_module = BehavioralFacade(dds=dds_pubsub, logger=logger, policy=policy)

    dds_pubsub.subscribe(TRAJECTORY_PARAMS_READER_TOPIC, behavioral_publish_mock)
    # dds_pubsub.subscribe(STATE_PUBLISH_TOPIC, state_module)

    behavioral_planner_module.start()
    behavioral_planner_module.periodic_action()

    # behavioral_publish_mock.periodic_action()

    behavioral_publish_mock.assert_called_once()
