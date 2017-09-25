import pickle
from logging import Logger

import numpy as np
from mapping.src.model.naive_cache_map import NaiveCacheMap

from common_data.dds.python.Communication.ddspubsub import DdsPubSub
from decision_making.paths import Paths
from decision_making.src.global_constants import *
from decision_making.src.manager.dm_manager import DmManager
from decision_making.src.manager.dm_process import DmProcess
from decision_making.src.manager.dm_trigger import *
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.planning.behavioral.behavioral_facade import BehavioralFacade
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState
from decision_making.src.planning.behavioral.default_policy import DefaultPolicy
from decision_making.src.planning.behavioral.default_policy_config import DefaultPolicyConfig
from decision_making.src.planning.navigation.navigation_facade import NavigationFacade
from decision_making.src.planning.trajectory.optimal_control.werling_planner import WerlingPlanner
from decision_making.src.planning.trajectory.trajectory_planning_facade import TrajectoryPlanningFacade
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.state.state import EgoState, ObjectSize, RoadLocalization, OccupancyState
from decision_making.src.state.state_module import StateModule
from rte.python.logger.AV_logger import AV_Logger

NAVIGATION_PLAN = NavigationPlanMsg(np.array([20]))


class NavigationFacadeMock(NavigationFacade):
    def __init__(self, dds: DdsPubSub, logger: Logger, plan: NavigationPlanMsg):
        super().__init__(dds=dds, logger=logger, handler=None)
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
        dds = DdsPubSub(STATE_MODULE_DDS_PARTICIPANT, DECISION_MAKING_DDS_FILE)
        map_model = pickle.load(open(Paths.get_resource_absolute_path_filename(MAP_RESOURCE_FILE_NAME), "rb"))
        map_api = NaiveCacheMap(map_model, logger)
        default_occupancy_state = OccupancyState(0, np.array([[1.1, 1.1, 0.1]], dtype=np.float),
                                                 np.array([0.1], dtype=np.float))
        state_module = StateModule(dds, logger, map_api, default_occupancy_state, None, None)
        return state_module

    @staticmethod
    def create_navigation_planner() -> NavigationFacade:
        logger = AV_Logger.get_logger(NAVIGATION_PLANNING_NAME_FOR_LOGGING)
        dds = DdsPubSub(NAVIGATION_PLANNER_DDS_PARTICIPANT, DECISION_MAKING_DDS_FILE)
        # TODO: fill navigation planning handlers
        # navigator = NavigationPlannerMock(plan)
        # navigation_module = NavigationFacadeMock(dds=dds, logger=logger, handler=navigator)
        navigation_module = NavigationFacadeMock(dds=dds, logger=logger, plan=NAVIGATION_PLAN)
        return navigation_module

    @staticmethod
    def create_behavioral_planner() -> BehavioralFacade:
        logger = AV_Logger.get_logger(BEHAVIORAL_PLANNING_NAME_FOR_LOGGING)
        dds = DdsPubSub(BEHAVIORAL_PLANNER_DDS_PARTICIPANT, DECISION_MAKING_DDS_FILE)
        # TODO: fill the policy
        map_model = pickle.load(open(Paths.get_resource_absolute_path_filename(MAP_RESOURCE_FILE_NAME), "rb"))
        map_api = NaiveCacheMap(map_model, logger)
        policy_config = DefaultPolicyConfig()
        policy = DefaultPolicy(logger, policy_config)

        init_navigation_plan = NavigationPlanMsg(np.array([]))
        init_ego_state = EgoState(0, None, 0.0, 0.0, 0.0, 0.0, ObjectSize(0.0, 0.0, 0.0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                  RoadLocalization(0, 0, 0.0, 0.0, 0.0, 0.0))

        behavioral_state = BehavioralState(logger, map_api, init_navigation_plan, init_ego_state, [])
        behavioral_module = BehavioralFacade(dds=dds, logger=logger, policy=policy, behavioral_state=behavioral_state)
        return behavioral_module

    @staticmethod
    def create_trajectory_planner() -> TrajectoryPlanningFacade:
        logger = AV_Logger.get_logger(TRAJECTORY_PLANNING_NAME_FOR_LOGGING)
        dds = DdsPubSub(TRAJECTORY_PLANNER_DDS_PARTICIPANT, DECISION_MAKING_DDS_FILE)
        # TODO: fill the strategy handlers
        planner = WerlingPlanner(logger)
        strategy_handlers = {TrajectoryPlanningStrategy.HIGHWAY: planner,
                             TrajectoryPlanningStrategy.PARKING: planner,
                             TrajectoryPlanningStrategy.TRAFFIC_JAM: planner}
        trajectory_planning_module = TrajectoryPlanningFacade(dds=dds, logger=logger,
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
    manager = DmManager(logger, modules_list)
    manager.start_modules()
    try:
        manager.wait_for_submodules()
    except KeyboardInterrupt:
        pass
    finally:
        manager.stop_modules()


main()
