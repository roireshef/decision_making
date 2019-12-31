import os

from decision_making.src.global_constants import ROUTE_PLANNING_NAME_FOR_LOGGING, \
    BEHAVIORAL_PLANNING_NAME_FOR_LOGGING, \
    TRAJECTORY_PLANNING_NAME_FOR_LOGGING, \
    DM_MANAGER_NAME_FOR_LOGGING, BEHAVIORAL_PLANNING_MODULE_PERIOD, TRAJECTORY_PLANNING_MODULE_PERIOD, \
    ROUTE_PLANNING_MODULE_PERIOD
from decision_making.src.infra.pubsub import PubSub
from decision_making.src.manager.dm_manager import DmManager
from decision_making.src.manager.dm_process import DmProcess
from decision_making.src.manager.dm_trigger import DmTriggerType
from decision_making.src.planning.behavioral.behavioral_planning_facade import BehavioralPlanningFacade
from decision_making.src.planning.behavioral.state.driver_initiated_motion_state import DIM_States
from decision_making.src.planning.behavioral.state.lane_change_state import LaneChangeStatus
from decision_making.src.planning.behavioral.visualization.dim_and_lcod_visualizer import DIMAndLCoDVisualizer
from decision_making.src.planning.route.backpropagating_route_planner import BackpropagatingRoutePlanner
from decision_making.src.planning.route.route_planning_facade import RoutePlanningFacade
from decision_making.src.planning.trajectory.trajectory_planning_facade import TrajectoryPlanningFacade
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.planning.trajectory.werling_planner import WerlingPlanner
from decision_making.src.prediction.ego_aware_prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.utils.async_visualizer import DummyVisualizer
from rte.python.logger.AV_logger import AV_Logger
from rte.python.os import catch_interrupt_signals
from rte.python.parser import av_argument_parser

AV_Logger.init_group("PLAN")

RUN_STATE_MACHINE_VISUALIZER = True


class DmInitialization:
    """
    This class contains the module initializations
    """
    @staticmethod
    def create_route_planner() -> RoutePlanningFacade:
        logger = AV_Logger.get_logger(ROUTE_PLANNING_NAME_FOR_LOGGING)

        pubsub = PubSub()

        planner = BackpropagatingRoutePlanner()

        route_planning_module = RoutePlanningFacade(pubsub=pubsub, logger=logger, route_planner=planner)
        return route_planning_module

    @staticmethod
    def create_behavioral_planner() -> BehavioralPlanningFacade:
        logger = AV_Logger.get_logger(BEHAVIORAL_PLANNING_NAME_FOR_LOGGING)

        pubsub = PubSub()

        # since it is sent to process from outside, the queue must be defined as a global variable
        global visualizer_queue

        behavioral_module = BehavioralPlanningFacade(pubsub=pubsub, logger=logger, last_trajectory=None,
                                                     visualizer_queue=visualizer_queue)
        return behavioral_module

    @staticmethod
    def create_trajectory_planner() -> TrajectoryPlanningFacade:
        logger = AV_Logger.get_logger(TRAJECTORY_PLANNING_NAME_FOR_LOGGING)

        pubsub = PubSub()

        predictor = RoadFollowingPredictor(logger)

        planner = WerlingPlanner(logger, predictor)
        strategy_handlers = {TrajectoryPlanningStrategy.HIGHWAY: planner,
                             TrajectoryPlanningStrategy.PARKING: planner,
                             TrajectoryPlanningStrategy.TRAFFIC_JAM: planner}

        trajectory_planning_module = TrajectoryPlanningFacade(pubsub=pubsub, logger=logger,
                                                              strategy_handlers=strategy_handlers)
        return trajectory_planning_module


if __name__ == '__main__':
    av_argument_parser.parse_arguments()
    # register termination signal handler
    logger = AV_Logger.get_logger(DM_MANAGER_NAME_FOR_LOGGING)
    logger.debug('%d: (DM main) registered signal handler', os.getpid())
    catch_interrupt_signals()

    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    # instantiate real state machine visualizer if RUN_STATE_MACHINE_VISUALIZER is True, otherwise a dummy one
    visualizer = DIMAndLCoDVisualizer() if RUN_STATE_MACHINE_VISUALIZER else DummyVisualizer()
    visualizer_queue = visualizer.queue
    visualizer.start()

    # put default values in the queue
    visualizer.append(DIM_States.DISABLED)
    visualizer.append(LaneChangeStatus.PENDING)

    modules_list = \
        [
            DmProcess(lambda: DmInitialization.create_route_planner(),
                      trigger_type=DmTriggerType.DM_TRIGGER_PERIODIC,
                      trigger_args={'period': ROUTE_PLANNING_MODULE_PERIOD},
                      name='RP'),

            DmProcess(lambda: DmInitialization.create_behavioral_planner(),
                      trigger_type=DmTriggerType.DM_TRIGGER_PERIODIC,
                      trigger_args={'period': BEHAVIORAL_PLANNING_MODULE_PERIOD},
                      name='BP'),

            DmProcess(lambda: DmInitialization.create_trajectory_planner(),
                      trigger_type=DmTriggerType.DM_TRIGGER_PERIODIC,
                      trigger_args={'period': TRAJECTORY_PLANNING_MODULE_PERIOD},
                      name='TP')
        ]

    manager = DmManager(logger, modules_list)
    manager.start_modules()
    try:
        manager.wait_for_submodules()
    except KeyboardInterrupt:
        logger.info('%d: (DM main) interrupted by signal', os.getpid())
        pass
    finally:
        manager.stop_modules()
        if RUN_STATE_MACHINE_VISUALIZER:
            visualizer.stop()
