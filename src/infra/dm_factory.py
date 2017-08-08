from enum import Enum
from common_data.dds.python.Communication.ddspubsub import DdsPubSub
from rte.python.logger.AV_logger import AV_Logger
from decision_making.src.global_constants import *
from decision_making.src.infra.dm_module import DmModule
from decision_making.src.planning.behavioral.behavioral_facade import BehavioralFacade
from decision_making.src.planning.trajectory.trajectory_planning_facade import TrajectoryPlanningFacade
from decision_making.src.state.state_module import StateModule
from decision_making.src.planning.behavioral.policy import DefaultPolicy
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState


class DmModulesEnum(Enum):
    DM_MODULE_STATE = 0
    DM_MODULE_BEHAVIORAL_PLANNER = 1
    DM_MODULE_TRAJECTORY_PLANNER = 2


class DmModuleFactory:

    @staticmethod
    def create_dm_module(module_enum: DmModulesEnum) -> DmModule:
        if module_enum == DmModulesEnum.DM_MODULE_STATE:
            return DmModuleFactory.__create_state_module()
        elif module_enum == DmModulesEnum.DM_MODULE_BEHAVIORAL_PLANNER:
            return DmModuleFactory.__create_behavioral_planner()
        elif module_enum == DmModulesEnum.DM_MODULE_TRAJECTORY_PLANNER:
            return DmModuleFactory.__create_trajectory_planner()
        else:
            raise ValueError("received unknown DM Module {}".format(module_enum))

    @staticmethod
    def __create_state_module() -> StateModule:
        logger = AV_Logger.get_logger(STATE_MODULE_NAME_FOR_LOGGING)
        dds = DdsPubSub(STATE_MODULE_DDS_PARTICIPANT, DECISION_MAKING_DDS_FILE)
        state_module = StateModule(dds, logger)
        return state_module

    @staticmethod
    def __create_behavioral_planner() -> BehavioralFacade:
        logger = AV_Logger.get_logger(BEHAVIORAL_PLANNING_NAME_FOR_LOGGING)
        dds = DdsPubSub(BEHAVIORAL_PLANNER_DDS_PARTICIPANT, DECISION_MAKING_DDS_FILE)
        # TODO: fill the policy
        policy_params = dict()
        policy = DefaultPolicy(policy_params)
        behavioral_state = BehavioralState()
        behavioral_module = BehavioralFacade(dds=dds, logger=logger, policy=policy,
                                             behavioral_state=behavioral_state)
        return behavioral_module

    @staticmethod
    def __create_trajectory_planner() -> TrajectoryPlanningFacade:
        logger = AV_Logger.get_logger(TRAJECTORY_PLANNING_NAME_FOR_LOGGING)
        dds = DdsPubSub(TRAJECTORY_PLANNER_DDS_PARTICIPANT, DECISION_MAKING_DDS_FILE)
        # TODO: fill the strategy handlers
        strategy_handlers = dict()
        trajectory_planning_module = TrajectoryPlanningFacade(dds=dds, logger=logger,
                                                              strategy_handlers=strategy_handlers)
        return trajectory_planning_module

