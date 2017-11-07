from logging import Logger

import numpy as np

from decision_making.src.global_constants import STATE_MODULE_NAME_FOR_LOGGING, STATE_MODULE_DDS_PARTICIPANT, \
    DECISION_MAKING_DDS_FILE, NAVIGATION_PLANNING_NAME_FOR_LOGGING, NAVIGATION_PLANNER_DDS_PARTICIPANT, \
    BEHAVIORAL_PLANNING_NAME_FOR_LOGGING, BEHAVIORAL_PLANNER_DDS_PARTICIPANT, TRAJECTORY_PLANNING_NAME_FOR_LOGGING, \
    TRAJECTORY_PLANNER_DDS_PARTICIPANT, BEHAVIORAL_PLANNING_MODULE_PERIOD, TRAJECTORY_PLANNING_MODULE_PERIOD, \
    DM_MANAGER_NAME_FOR_LOGGING
from decision_making.src.manager.dm_trigger import DmTriggerType
from decision_making.src.planning.behavioral.policies.november_demo_semantic_policy import NovDemoPolicy
from decision_making.src.prediction.road_following_predictor import RoadFollowingPredictor
from mapping.src.service.map_service import MapService

from common_data.dds.python.Communication.ddspubsub import DdsPubSub
from decision_making.src.manager.dm_manager import DmManager
from decision_making.src.manager.dm_process import DmProcess
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.planning.behavioral.behavioral_facade import BehavioralFacade
from decision_making.src.planning.behavioral.policies.default_policy import DefaultPolicy, DefaultBehavioralState
from decision_making.src.planning.behavioral.policies.default_policy_config import DefaultPolicyConfig
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
        MapService.initialize()
        map_api = MapService.get_instance()
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
        # Init map
        MapService.initialize()
        map_api = MapService.get_instance()

        # Init states
        init_navigation_plan = NavigationPlanMsg(np.array([]))
        init_ego_state = EgoState(0, None, 0.0, 0.0, 0.0, 0.0, ObjectSize(0.0, 0.0, 0.0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                  RoadLocalization(0, 0, 0.0, 0.0, 0.0, 0.0))

        # Init policy
        # behavioral_state = DefaultBehavioralState(logger, map_api, init_navigation_plan, init_ego_state, [])
        # policy_config = DefaultPolicyConfig()
        # policy = DefaultPolicy(logger, policy_config, behavioral_state, None, map_api)

        # NOV DEMO POLICY
        predictor = RoadFollowingPredictor(map_api)
        policy = NovDemoPolicy(logger=logger, policy_config=None, predictor=predictor, map_api=map_api)

        behavioral_module = BehavioralFacade(dds=dds, logger=logger, policy=policy)
        return behavioral_module

    @staticmethod
    def create_trajectory_planner() -> TrajectoryPlanningFacade:
        logger = AV_Logger.get_logger(TRAJECTORY_PLANNING_NAME_FOR_LOGGING)
        dds = DdsPubSub(TRAJECTORY_PLANNER_DDS_PARTICIPANT, DECISION_MAKING_DDS_FILE)

        # Init map
        MapService.initialize()
        map_api = MapService.get_instance()
        init_navigation_plan = NAVIGATION_PLAN

        predictor = RoadFollowingPredictor(map_api)

        # TODO: fill the strategy handlers
        planner = WerlingPlanner(logger, predictor)
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
