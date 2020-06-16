import sys
import ctypes

from rte.python.process.process_manager import IProcessManager
from rte.python.process.process_skeleton import ProcessSkeleton

PROC_NAME = b"Planning"

if sys.platform.startswith('linux'):
    libc = ctypes.cdll.LoadLibrary('libc.so.6')
    libc.prctl(15, PROC_NAME, 0, 0, 0)


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
from rte.python.logger.AV_logger import AV_Logger
from rte.python.parser import av_argument_parser
from decision_making.src.utils.dummy_queue import DummyQueue

AV_Logger.init_group("PLAN")

RUN_STATE_MACHINE_VISUALIZER = False


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

        # queue is sent to process from outside, it must be defined as a global variable, which is populated below
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


class DMManagerAdapter(IProcessManager):
    def __init__(self):
        self.manager = None
        self.visualizer = None

    def register_command_line_args(self, parser):
        pass

    def init(self, arguments):
        logger = AV_Logger.get_logger(DM_MANAGER_NAME_FOR_LOGGING)

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

        self.manager = DmManager(logger, modules_list)

    def start_running(self):
        # instantiate real state machine visualizer and get its queue, or define a DummyQueue that implements the queue
        # interface and does nothing. Note that the visualizer_queue is read as a global variable above. This is where
        # it is populated in the first place
        if RUN_STATE_MACHINE_VISUALIZER:
            self.visualizer = DIMAndLCoDVisualizer()
            self.visualizer.start()

            visualizer_queue = self.visualizer.queue

            # put default values in the queue
            self.visualizer.append(DIM_States.DISABLED)
            self.visualizer.append(LaneChangeStatus.PENDING)
        else:
            visualizer_queue = DummyQueue()

        self.manager.start_modules()

    def wait_loop(self):
        self.manager.wait_for_submodules()

    def terminate(self):
        self.manager.stop_modules()
        if RUN_STATE_MACHINE_VISUALIZER:
            self.visualizer.stop()

    def cleanup(self):
        pass


if __name__ == '__main__':
    # register termination signal handler

    m = DMManagerAdapter()
    p = ProcessSkeleton("planning", m)

    p.start()
