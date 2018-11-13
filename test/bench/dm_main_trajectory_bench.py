from os import getpid

import numpy as np

from common_data.interface.py.pubsub.Rte_Types_pubsub_topics import PubSubMessageTypes
from common_data.src.communication.pubsub.pubsub_factory import create_pubsub
from decision_making.src import global_constants
from decision_making.src.dm_main import DmInitialization
from decision_making.src.global_constants import BEHAVIORAL_PLANNING_MODULE_PERIOD, TRAJECTORY_PLANNING_MODULE_PERIOD, \
    DM_MANAGER_NAME_FOR_LOGGING, TRAJECTORY_PLANNING_NAME_FOR_LOGGING, TRAJECTORY_TIME_RESOLUTION, \
    FIXED_TRAJECTORY_PLANNER_SLEEP_STD, FIXED_TRAJECTORY_PLANNER_SLEEP_MEAN, STATE_MODULE_NAME_FOR_LOGGING
from decision_making.src.manager.dm_manager import DmManager
from decision_making.src.manager.dm_process import DmProcess
from decision_making.src.manager.dm_trigger import DmTriggerType
from decision_making.src.planning.trajectory.fixed_trajectory_planner import FixedTrajectoryPlanner
from decision_making.src.planning.trajectory.trajectory_planning_facade import TrajectoryPlanningFacade
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.planning.types import C_Y
from decision_making.src.prediction.action_unaware_prediction.physical_time_alignment_predictor import \
    PhysicalTimeAlignmentPredictor
from decision_making.src.prediction.ego_aware_prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.state.state import OccupancyState
from decision_making.src.state.state_module import StateModule
from decision_making.test import constants
from decision_making.test.constants import TP_MOCK_FIXED_TRAJECTORY_FILENAME
from decision_making.test.utils_for_tests import Utils
from mapping.src.service.map_service import MapService
from rte.python.logger.AV_logger import AV_Logger
from rte.python.os import catch_interrupt_signals


class DmMockInitialization:

    @staticmethod

    #The purpose of this initialization is to generate a state module holding an initial empty list of dyanmic object.
    #The purpose here is to continuousely publish localization (as long as it is available from the IMU) wihtout waiting
    #for a dynamic object update.
    def create_state_module() -> StateModule:
        logger = AV_Logger.get_logger(STATE_MODULE_NAME_FOR_LOGGING)
        pubsub = create_pubsub(PubSubMessageTypes)
        MapService.initialize()
        #TODO: figure out if we want to use OccupancyState at all
        default_occupancy_state = OccupancyState(0, np.array([[1.1, 1.1, 0.1]], dtype=np.float),
                                                 np.array([0.1], dtype=np.float))

        state_module = StateModule(pubsub, logger, default_occupancy_state, [], None)
        return state_module

    @staticmethod
    def create_trajectory_planner(fixed_trajectory_file: str = None) -> TrajectoryPlanningFacade:
        logger = AV_Logger.get_logger(TRAJECTORY_PLANNING_NAME_FOR_LOGGING)
        pubsub = create_pubsub(PubSubMessageTypes)
        # Init map
        MapService.initialize()
        predictor = RoadFollowingPredictor(logger)

        if fixed_trajectory_file is None:
            fixed_trajectory = Utils.read_trajectory(TP_MOCK_FIXED_TRAJECTORY_FILENAME)
        else:
            fixed_trajectory = Utils.read_trajectory(fixed_trajectory_file)

        step_size = TRAJECTORY_PLANNING_MODULE_PERIOD / TRAJECTORY_TIME_RESOLUTION
        planner = FixedTrajectoryPlanner(logger, predictor, fixed_trajectory, step_size,
                                         fixed_trajectory[0, :(C_Y+1)],
                                         FIXED_TRAJECTORY_PLANNER_SLEEP_STD,
                                         FIXED_TRAJECTORY_PLANNER_SLEEP_MEAN)

        strategy_handlers = {TrajectoryPlanningStrategy.HIGHWAY: planner,
                             TrajectoryPlanningStrategy.PARKING: planner,
                             TrajectoryPlanningStrategy.TRAFFIC_JAM: planner}

        short_time_predictor = PhysicalTimeAlignmentPredictor(logger)
        trajectory_planning_module = TrajectoryPlanningFacade(pubsub=pubsub, logger=logger,
                                                              strategy_handlers=strategy_handlers,
                                                              short_time_predictor=short_time_predictor)
        return trajectory_planning_module



def main(fixed_trajectory_file: str = None):
    """
    initializes DM planning pipeline. for switching between BP/TP impl./mock make sure to comment out the relevant
    instantiation in modules_list.
    """
    modules_list = \
        [
            DmProcess(DmInitialization.create_navigation_planner,
                      trigger_type=DmTriggerType.DM_TRIGGER_PERIODIC,
                      trigger_args={'period': BEHAVIORAL_PLANNING_MODULE_PERIOD}),

            DmProcess(DmMockInitialization.create_state_module,
                      trigger_type=DmTriggerType.DM_TRIGGER_NONE,
                      trigger_args={}),

            DmProcess(DmInitialization.create_behavioral_planner,
                      trigger_type=DmTriggerType.DM_TRIGGER_PERIODIC,
                      trigger_args={'period': BEHAVIORAL_PLANNING_MODULE_PERIOD}),

            DmProcess(lambda: DmMockInitialization.create_trajectory_planner(fixed_trajectory_file=fixed_trajectory_file),
                      trigger_type=DmTriggerType.DM_TRIGGER_PERIODIC,
                      trigger_args={'period': TRAJECTORY_PLANNING_MODULE_PERIOD})
        ]
    logger = AV_Logger.get_logger(DM_MANAGER_NAME_FOR_LOGGING)
    logger.debug('%d: (main) registered signal handler', getpid())
    logger.info("DM Global Constants: %s", {k: v for k, v in global_constants.__dict__.items() if k.isupper()})
    logger.info("DM Test Constants: %s", {k: v for k, v in constants.__dict__.items() if k.isupper()})

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


if __name__ == '__main__':
    main()
