from logging import Logger
from os import getpid

import numpy as np
import os

from decision_making.paths import Paths
from decision_making.src.planning.behavioral.evaluators.single_lane_action_spec_evaluator import \
    SingleLaneActionSpecEvaluator

from common_data.interface.py.pubsub.Rte_Types_pubsub_topics import PubSubMessageTypes
from common_data.src.communication.pubsub.pubsub import PubSub
from common_data.src.communication.pubsub.pubsub_factory import create_pubsub
from decision_making.src.global_constants import STATE_MODULE_NAME_FOR_LOGGING, \
    NAVIGATION_PLANNING_NAME_FOR_LOGGING, \
    ROUTE_PLANNING_NAME_FOR_LOGGING, \
    BEHAVIORAL_PLANNING_NAME_FOR_LOGGING, \
    TRAJECTORY_PLANNING_NAME_FOR_LOGGING, \
    DM_MANAGER_NAME_FOR_LOGGING, BEHAVIORAL_PLANNING_MODULE_PERIOD, TRAJECTORY_PLANNING_MODULE_PERIOD
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
from decision_making.src.planning.behavioral.evaluators.zero_value_approximator import ZeroValueApproximator
from decision_making.src.planning.behavioral.filtering.action_spec_filter_bank import FilterIfNone
from decision_making.src.planning.behavioral.filtering.action_spec_filtering import ActionSpecFiltering
from decision_making.src.planning.behavioral.planner.single_step_behavioral_planner import SingleStepBehavioralPlanner
from decision_making.src.planning.navigation.navigation_facade import NavigationFacade
from decision_making.src.planning.route.route_planning_facade import RoutePlanningFacade
from decision_making.src.planning.route.dual_cost_route_planner import DualCostRoutePlanner
from decision_making.src.planning.trajectory.trajectory_planning_facade import TrajectoryPlanningFacade
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.planning.trajectory.werling_planner import WerlingPlanner
from decision_making.src.prediction.ego_aware_prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.state.state_module import StateModule
from mapping.src.service.map_service import MapService
from rte.python.logger.AV_logger import AV_Logger
from rte.python.os import catch_interrupt_signals

# TODO: move this into config?
NAVIGATION_PLAN = NavigationPlanMsg(np.array([231800832, 5007343616,  238944256, 3052470272, 3054829568,
                                              5751373824,  574488576, 5035655168, 5751111680, 3365928960,
                                              5751046144, 5742395392, 5727059968, 5744885760, 5750390784,
                                              648347648,  648413184, 5750128640, 5072355328, 5750194176,
                                              1701904384,  659816448, 5715460096,  676331520,  676462592,
                                              9135521792, 5750063104, 2683895808,  660144128,  659685376,
                                              5749604352, 2851864576, 5752094720, 5766250496, 5766905856,
                                              5771821056,  690487296, 5772935168, 5774770176, 5779750912,
                                              689373184,  683671552,

                                              231800832, 5007343616, 238944256, 3052470272, 3054829568,
                                              5751373824, 574488576, 5035655168, 5751111680, 3365928960,
                                              5751046144, 5742395392, 5727059968, 5744885760, 5750390784,
                                              648347648, 648413184, 5750128640, 5072355328, 5750194176,
                                              1701904384, 659816448, 5715460096, 676331520, 676462592,
                                              9135521792, 5750063104, 2683895808, 660144128, 659685376,
                                              5749604352, 2851864576, 5752094720, 5766250496, 5766905856,
                                              5771821056, 690487296, 5772935168, 5774770176, 5779750912,
                                              689373184, 683671552]))

NAVIGATION_PLAN_PG = NavigationPlanMsg(np.array(range(20, 30)))  # 20 for Ayalon PG
DEFAULT_MAP_FILE = Paths.get_repo_path() + '/../common_data/maps/PG_split.bin'


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
        state_module = StateModule(pubsub, logger, None)
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
    def create_route_planner(map_file: str=DEFAULT_MAP_FILE,) -> RoutePlanningFacade:
        logger = AV_Logger.get_logger(ROUTE_PLANNING_NAME_FOR_LOGGING)
        pubsub = create_pubsub(PubSubMessageTypes)
        # MapService should be initialized in each process according to the given map_file
        MapService.initialize(map_file)

        planner = DualCostRoutePlanner()

        route_planning_module = RoutePlanningFacade(pubsub=pubsub, logger=logger, planner=planner)
        return route_planning_module

    @staticmethod
    def create_behavioral_planner(map_file: str=DEFAULT_MAP_FILE) -> BehavioralPlanningFacade:
        logger = AV_Logger.get_logger(BEHAVIORAL_PLANNING_NAME_FOR_LOGGING)
        pubsub = create_pubsub(PubSubMessageTypes)
        # MapService should be initialized in each process according to the given map_file
        MapService.initialize(map_file)

        predictor = RoadFollowingPredictor(logger)

        action_space = ActionSpaceContainer(logger, [StaticActionSpace(logger, DEFAULT_STATIC_RECIPE_FILTERING),
                                                     DynamicActionSpace(logger, predictor,
                                                                        DEFAULT_DYNAMIC_RECIPE_FILTERING)])

        recipe_evaluator = None
        action_spec_evaluator = SingleLaneActionSpecEvaluator(logger)  # RuleBasedActionSpecEvaluator(logger)
        value_approximator = ZeroValueApproximator(logger)

        action_spec_filtering = ActionSpecFiltering(filters=[FilterIfNone()], logger=logger)
        planner = SingleStepBehavioralPlanner(action_space, recipe_evaluator, action_spec_evaluator,
                                              action_spec_filtering, value_approximator, predictor, logger)

        behavioral_module = BehavioralPlanningFacade(pubsub=pubsub, logger=logger,
                                                     behavioral_planner=planner, last_trajectory=None)
        return behavioral_module

    @staticmethod
    def create_trajectory_planner(map_file: str=DEFAULT_MAP_FILE) -> TrajectoryPlanningFacade:
        logger = AV_Logger.get_logger(TRAJECTORY_PLANNING_NAME_FOR_LOGGING)
        pubsub = create_pubsub(PubSubMessageTypes)
        # MapService should be initialized in each process according to the given map_file
        MapService.initialize(map_file)

        predictor = RoadFollowingPredictor(logger)

        planner = WerlingPlanner(logger, predictor)
        strategy_handlers = {TrajectoryPlanningStrategy.HIGHWAY: planner,
                             TrajectoryPlanningStrategy.PARKING: planner,
                             TrajectoryPlanningStrategy.TRAFFIC_JAM: planner}

        trajectory_planning_module = TrajectoryPlanningFacade(pubsub=pubsub, logger=logger,
                                                              strategy_handlers=strategy_handlers)
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

            DmProcess(lambda: DmInitialization.create_route_planner(DEFAULT_MAP_FILE),
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
