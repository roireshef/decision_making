from logging import Logger
from os import getpid

import numpy as np

from common_data.src.communication.pubsub.pubsub_factory import create_pubsub
from common_data.src.communication.pubsub.pubsub import PubSub
from common_data.lcm.python.Communication.lcmpubsub import LcmPubSub
from common_data.lcm.config import config_defs
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
from decision_making.src.planning.behavioral.behavioral_facade import BehavioralFacade
from decision_making.src.planning.behavioral.policies.semantic_actions_grid_policy import SemanticActionsGridPolicy
from decision_making.src.planning.navigation.navigation_facade import NavigationFacade
from decision_making.src.planning.trajectory.optimal_control.werling_planner import WerlingPlanner
from decision_making.src.planning.trajectory.trajectory_planning_facade import TrajectoryPlanningFacade
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.state.state import OccupancyState
from decision_making.src.state.state_module import StateModule
from mapping.src.service.map_service import MapService
from rte.python.logger.AV_logger import AV_Logger
from rte.python.os import catch_interrupt_signals

# TODO: move this into test package
NAVIGATION_PLAN = NavigationPlanMsg(np.array([20]))
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
    def create_state_module() -> StateModule:
        logger = AV_Logger.get_logger(STATE_MODULE_NAME_FOR_LOGGING)
        pubsub = create_pubsub(config_defs.LCM_SOCKET_CONFIG, LcmPubSub)
        MapService.initialize()
        default_occupancy_state = OccupancyState(0, np.array([[1.1, 1.1, 0.1]], dtype=np.float),
                                                 np.array([0.1], dtype=np.float))
        state_module = StateModule(pubsub, logger, default_occupancy_state, None, None)
        return state_module

    @staticmethod
    def create_navigation_planner() -> NavigationFacade:
        logger = AV_Logger.get_logger(NAVIGATION_PLANNING_NAME_FOR_LOGGING)
        pubsub = create_pubsub(config_defs.LCM_SOCKET_CONFIG, LcmPubSub)

        # TODO: fill navigation planning handlers
        # navigator = NavigationPlannerMock(plan)
        # navigation_module = NavigationFacadeMock(pubsub=pubsub, logger=logger, handler=navigator)
        navigation_module = NavigationFacadeMock(pubsub=pubsub, logger=logger, plan=NAVIGATION_PLAN)
        return navigation_module

    @staticmethod
    def create_behavioral_planner() -> BehavioralFacade:
        logger = AV_Logger.get_logger(BEHAVIORAL_PLANNING_NAME_FOR_LOGGING)
        pubsub = create_pubsub(config_defs.LCM_SOCKET_CONFIG, LcmPubSub)
        # TODO: fill the policy
        # Init map
        MapService.initialize()
        predictor = RoadFollowingPredictor(logger)
        policy = SemanticActionsGridPolicy(logger=logger, predictor=predictor)

        behavioral_module = BehavioralFacade(pubsub=pubsub, logger=logger, policy=policy)
        return behavioral_module

    @staticmethod
    def create_trajectory_planner() -> TrajectoryPlanningFacade:
        logger = AV_Logger.get_logger(TRAJECTORY_PLANNING_NAME_FOR_LOGGING)
        pubsub = create_pubsub(config_defs.LCM_SOCKET_CONFIG, LcmPubSub)

        # Init map
        MapService.initialize()
        predictor = RoadFollowingPredictor(logger)

        # TODO: fill the strategy handlers
        planner = WerlingPlanner(logger, predictor)
        strategy_handlers = {TrajectoryPlanningStrategy.HIGHWAY: planner,
                             TrajectoryPlanningStrategy.PARKING: planner,
                             TrajectoryPlanningStrategy.TRAFFIC_JAM: planner}

        trajectory_planning_module = TrajectoryPlanningFacade(pubsub=pubsub, logger=logger,
                                                              strategy_handlers=strategy_handlers)
        return trajectory_planning_module


def main():
    modules_list = \
        [
            DmProcess(DmInitialization.create_navigation_planner,
                      trigger_type=DmTriggerType.DM_TRIGGER_PERIODIC,
                      trigger_args={'period': BEHAVIORAL_PLANNING_MODULE_PERIOD}),

            DmProcess(DmInitialization.create_state_module,
                      trigger_type=DmTriggerType.DM_TRIGGER_NONE,
                      trigger_args={}),

            DmProcess(DmInitialization.create_behavioral_planner,
                      trigger_type=DmTriggerType.DM_TRIGGER_PERIODIC,
                      trigger_args={'period': BEHAVIORAL_PLANNING_MODULE_PERIOD}),

            DmProcess(DmInitialization.create_trajectory_planner,
                      trigger_type=DmTriggerType.DM_TRIGGER_PERIODIC,
                      trigger_args={'period': TRAJECTORY_PLANNING_MODULE_PERIOD})
        ]
    logger = AV_Logger.get_logger(DM_MANAGER_NAME_FOR_LOGGING)
    logger.debug('%d: (main) registered signal handler', getpid())
    catch_interrupt_signals()
    manager = DmManager(logger, modules_list)
    manager.start_modules()
    try:
        manager.wait_for_submodules()
    except KeyboardInterrupt:
        logger.info('%d: (main) Interrupted by signal!', getpid())
        pass
    finally:
        manager.stop_modules()


main()
