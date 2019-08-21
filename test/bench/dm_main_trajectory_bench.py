from os import getpid

from decision_making.src import global_constants
from decision_making.src.dm_main import DmInitialization, DEFAULT_MAP_FILE
from decision_making.src.global_constants import BEHAVIORAL_PLANNING_MODULE_PERIOD, TRAJECTORY_PLANNING_MODULE_PERIOD, \
    DM_MANAGER_NAME_FOR_LOGGING, TRAJECTORY_PLANNING_NAME_FOR_LOGGING, TRAJECTORY_TIME_RESOLUTION, \
    FIXED_TRAJECTORY_PLANNER_SLEEP_STD, FIXED_TRAJECTORY_PLANNER_SLEEP_MEAN
from decision_making.src.infra.pubsub import PubSub
from decision_making.src.manager.dm_manager import DmManager
from decision_making.src.manager.dm_process import DmProcess
from decision_making.src.manager.dm_trigger import DmTriggerType
from decision_making.src.planning.trajectory.trajectory_planning_facade import TrajectoryPlanningFacade
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.planning.types import C_Y
from decision_making.src.prediction.ego_aware_prediction.road_following_predictor import RoadFollowingPredictor

from decision_making.test import constants
from decision_making.test.constants import TP_MOCK_FIXED_TRAJECTORY_FILENAME
from decision_making.src.planning.trajectory.fixed_trajectory_planner import FixedTrajectoryPlanner
from decision_making.test.utils_for_tests import Utils
from rte.python.logger.AV_logger import AV_Logger
from rte.python.os import catch_interrupt_signals
from decision_making.test.state.scene_dynamic_mock import SceneDynamicMock


class DmMockInitialization:

    @staticmethod
    def create_scene_dynamic_publisher() -> SceneDynamicMock:
        """
        The purpose of this initialization is to generate a scene dynamic mock holding an initial empty data
        :return:
        """
        logger = AV_Logger.get_logger(TRAJECTORY_PLANNING_NAME_FOR_LOGGING)

        pubsub = PubSub()

        scene_dynamic_mock = SceneDynamicMock(pubsub, logger, None)
        return scene_dynamic_mock

    @staticmethod
    def create_trajectory_planner(fixed_trajectory_file: str = None) -> TrajectoryPlanningFacade:
        logger = AV_Logger.get_logger(TRAJECTORY_PLANNING_NAME_FOR_LOGGING)

        pubsub = PubSub()

        predictor = RoadFollowingPredictor(logger)

        fixed_trajectory = Utils.read_trajectory(fixed_trajectory_file or TP_MOCK_FIXED_TRAJECTORY_FILENAME)

        step_size = TRAJECTORY_PLANNING_MODULE_PERIOD / TRAJECTORY_TIME_RESOLUTION
        planner = FixedTrajectoryPlanner(logger, predictor, fixed_trajectory, step_size,
                                         fixed_trajectory[0, :(C_Y+1)],
                                         FIXED_TRAJECTORY_PLANNER_SLEEP_STD,
                                         FIXED_TRAJECTORY_PLANNER_SLEEP_MEAN)

        strategy_handlers = {TrajectoryPlanningStrategy.HIGHWAY: planner,
                             TrajectoryPlanningStrategy.PARKING: planner,
                             TrajectoryPlanningStrategy.TRAFFIC_JAM: planner}

        trajectory_planning_module = TrajectoryPlanningFacade(pubsub=pubsub, logger=logger,
                                                              strategy_handlers=strategy_handlers)
        return trajectory_planning_module


def main(fixed_trajectory_file: str = None, map_file: str = DEFAULT_MAP_FILE):
    """
    initializes DM planning pipeline. for switching between BP/TP impl./mock make sure to comment out the relevant
    instantiation in modules_list.
    """
    modules_list = \
        [
            DmProcess(lambda: DmMockInitialization.create_scene_dynamic_publisher(),
                      trigger_type=DmTriggerType.DM_TRIGGER_PERIODIC,
                      trigger_args={'period': TRAJECTORY_PLANNING_MODULE_PERIOD}),

            DmProcess(lambda: DmInitialization.create_behavioral_planner(),
                      trigger_type=DmTriggerType.DM_TRIGGER_PERIODIC,
                      trigger_args={'period': BEHAVIORAL_PLANNING_MODULE_PERIOD}),

            DmProcess(lambda: DmMockInitialization.create_trajectory_planner(fixed_trajectory_file),
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
