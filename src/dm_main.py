from logging import Logger
from os import getpid

import os
import numpy as np

from common_data.interface.py.pubsub.Rte_Types_pubsub_topics import PubSubMessageTypes
from common_data.src.communication.pubsub.pubsub import PubSub
from common_data.src.communication.pubsub.pubsub_factory import create_pubsub
from decision_making.src.global_constants import STATE_MODULE_NAME_FOR_LOGGING, \
    NAVIGATION_PLANNING_NAME_FOR_LOGGING, \
    BEHAVIORAL_PLANNING_NAME_FOR_LOGGING, \
    BEHAVIORAL_PLANNING_MODULE_PERIOD, \
    TRAJECTORY_PLANNING_NAME_FOR_LOGGING, \
    TRAJECTORY_PLANNING_MODULE_PERIOD, \
    DM_MANAGER_NAME_FOR_LOGGING
from decision_making.src.manager.dm_manager import DmManager
from decision_making.src.manager.dm_process import DmProcess
from decision_making.src.manager.dm_trigger import DmTriggerType
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.planning.behavioral.action_space.action_space import ActionSpaceContainer
from decision_making.src.planning.behavioral.action_space.dynamic_action_space import DynamicActionSpace
from decision_making.src.planning.behavioral.action_space.static_action_space import StaticActionSpace
from decision_making.src.planning.behavioral.behavioral_planning_facade import BehavioralPlanningFacade
from decision_making.src.planning.behavioral.default_config import DEFAULT_DYNAMIC_RECIPE_FILTERING, \
    DEFAULT_STATIC_RECIPE_FILTERING
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
from decision_making.src.prediction.action_unaware_prediction.physical_time_alignment_predictor import \
    PhysicalTimeAlignmentPredictor
from decision_making.src.prediction.ego_aware_prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.state.state import OccupancyState
from decision_making.src.state.state_module import StateModule
from mapping.src.global_constants import DEFAULT_MAP_FILE
from mapping.src.service.map_service import MapService
from rte.python.logger.AV_logger import AV_Logger
from rte.python.os import catch_interrupt_signals

# TODO: move this into config?
NAVIGATION_PLAN = NavigationPlanMsg(np.array(range(20, 30)))  # 20 for Ayalon PG
DEFAULT_MAP_FILE = None  # os.environ['AVCODE_PATH'] + '/spav/common_data/maps/OvalMilford.bin'  # None for Ayalon PG


class NavigationFacadeMock(NavigationFacade):
    def __init__(self, pubsub: PubSub, logger: Logger, plan: NavigationPlanMsg):
        super().__init__(pubsub=pubsub, logger=logger, handler=None)
        self.plan = plan

    def _periodic_action_impl(self):
        self._publish_navigation_plan(self.plan)


class DmInitialization:
    """
    This class contains the module initializations
    """

    @staticmethod
    def create_state_module(map_file: str=DEFAULT_MAP_FILE) -> StateModule:
        logger = AV_Logger.get_logger(STATE_MODULE_NAME_FOR_LOGGING)
        pubsub = create_pubsub(PubSubMessageTypes)
        # MapService should be initialized in each process according to the given map_file
        MapService.initialize(map_file)
        # TODO: figure out if we want to use OccupancyState at all
        default_occupancy_state = OccupancyState(0, np.array([[1.1, 1.1, 0.1]], dtype=np.float),
                                                 np.array([0.1], dtype=np.float))
        state_module = StateModule(pubsub, logger, default_occupancy_state, None, None)
        return state_module

    @staticmethod
    def create_navigation_planner(map_file: str=DEFAULT_MAP_FILE, nav_plan: NavigationPlanMsg=NAVIGATION_PLAN) -> NavigationFacade:
        logger = AV_Logger.get_logger(NAVIGATION_PLANNING_NAME_FOR_LOGGING)
        pubsub = create_pubsub(PubSubMessageTypes)
        # MapService should be initialized in each process according to the given map_file
        MapService.initialize(map_file)

        navigation_module = NavigationFacadeMock(pubsub=pubsub, logger=logger, plan=nav_plan)
        return navigation_module

    @staticmethod
    def create_behavioral_planner(map_file: str=DEFAULT_MAP_FILE) -> BehavioralPlanningFacade:
        logger = AV_Logger.get_logger(BEHAVIORAL_PLANNING_NAME_FOR_LOGGING)
        pubsub = create_pubsub(PubSubMessageTypes)
        # MapService should be initialized in each process according to the given map_file
        MapService.initialize(map_file)

        predictor = RoadFollowingPredictor(logger)

        short_time_predictor = PhysicalTimeAlignmentPredictor(logger)

        action_space = ActionSpaceContainer(logger, [StaticActionSpace(logger, DEFAULT_STATIC_RECIPE_FILTERING),
                                                     DynamicActionSpace(logger, predictor,
                                                                        DEFAULT_DYNAMIC_RECIPE_FILTERING)])

        recipe_evaluator = None
        action_spec_evaluator = RuleBasedActionSpecEvaluator(logger)
        value_approximator = ZeroValueApproximator(logger)

        action_spec_filtering = ActionSpecFiltering(filters=[FilterIfNone()], logger=logger)
        planner = SingleStepBehavioralPlanner(action_space, recipe_evaluator, action_spec_evaluator,
                                              action_spec_filtering, value_approximator, predictor, logger)

        behavioral_module = BehavioralPlanningFacade(pubsub=pubsub, logger=logger,
                                                     behavioral_planner=planner,
                                                     short_time_predictor=short_time_predictor, last_trajectory=None)
        return behavioral_module

    @staticmethod
    def create_trajectory_planner(map_file: str=DEFAULT_MAP_FILE) -> TrajectoryPlanningFacade:
        logger = AV_Logger.get_logger(TRAJECTORY_PLANNING_NAME_FOR_LOGGING)
        pubsub = create_pubsub(PubSubMessageTypes)
        # MapService should be initialized in each process according to the given map_file
        MapService.initialize(map_file)

        predictor = RoadFollowingPredictor(logger)
        short_time_predictor = PhysicalTimeAlignmentPredictor(logger)

        planner = WerlingPlanner(logger, predictor)
        strategy_handlers = {TrajectoryPlanningStrategy.HIGHWAY: planner,
                             TrajectoryPlanningStrategy.PARKING: planner,
                             TrajectoryPlanningStrategy.TRAFFIC_JAM: planner}

        trajectory_planning_module = TrajectoryPlanningFacade(pubsub=pubsub, logger=logger,
                                                              strategy_handlers=strategy_handlers,
                                                              short_time_predictor=short_time_predictor)
        return trajectory_planning_module


def main():
    # register termination signal handler
    logger = AV_Logger.get_logger(DM_MANAGER_NAME_FOR_LOGGING)
    logger.debug('%d: (DM main) registered signal handler', getpid())
    catch_interrupt_signals()

    modules_list = \
        [
            DmProcess(lambda: DmInitialization.create_navigation_planner(DEFAULT_MAP_FILE),
                      trigger_type=DmTriggerType.DM_TRIGGER_PERIODIC,
                      trigger_args={'period': BEHAVIORAL_PLANNING_MODULE_PERIOD}),

            DmProcess(lambda: DmInitialization.create_state_module(DEFAULT_MAP_FILE),
                      trigger_type=DmTriggerType.DM_TRIGGER_NONE,
                      trigger_args={}),

            DmProcess(lambda: DmInitialization.create_behavioral_planner(DEFAULT_MAP_FILE),
                      trigger_type=DmTriggerType.DM_TRIGGER_PERIODIC,
                      trigger_args={'period': BEHAVIORAL_PLANNING_MODULE_PERIOD}),

            DmProcess(lambda: DmInitialization.create_trajectory_planner(DEFAULT_MAP_FILE),
                      trigger_type=DmTriggerType.DM_TRIGGER_PERIODIC,
                      trigger_args={'period': TRAJECTORY_PLANNING_MODULE_PERIOD})
        ]

    manager = DmManager(logger, modules_list)
    manager.start_modules()
    try:
        manager.wait_for_submodules()
    except KeyboardInterrupt:
        logger.info('%d: (DM main) interrupted by signal', getpid())
        pass
    finally:
        manager.stop_modules()


if __name__ == '__main__':
    main()
