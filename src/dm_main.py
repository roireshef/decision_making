from decision_making.src.global_constants import STATE_MODULE_NAME_FOR_LOGGING, \
    ROUTE_PLANNING_NAME_FOR_LOGGING, \
    BEHAVIORAL_PLANNING_NAME_FOR_LOGGING, \
    TRAJECTORY_PLANNING_NAME_FOR_LOGGING, \
    DM_MANAGER_NAME_FOR_LOGGING, BEHAVIORAL_PLANNING_MODULE_PERIOD, TRAJECTORY_PLANNING_MODULE_PERIOD, ROUTE_PLANNING_MODULE_PERIOD
from decision_making.paths import Paths
from decision_making.src.infra.pubsub import PubSub
from decision_making.src.manager.dm_manager import DmManager
from decision_making.src.manager.dm_process import DmProcess
from decision_making.src.manager.dm_trigger import DmTriggerType
from decision_making.src.planning.behavioral.action_space.action_space import ActionSpaceContainer
from decision_making.src.planning.behavioral.action_space.dynamic_action_space import DynamicActionSpace
from decision_making.src.planning.behavioral.action_space.static_action_space import StaticActionSpace
from decision_making.src.planning.behavioral.action_space.road_sign_action_space import RoadSignActionSpace
from decision_making.src.planning.behavioral.behavioral_planning_facade import BehavioralPlanningFacade
from decision_making.src.planning.behavioral.default_config import DEFAULT_DYNAMIC_RECIPE_FILTERING, \
    DEFAULT_STATIC_RECIPE_FILTERING, DEFAULT_ACTION_SPEC_FILTERING
from decision_making.src.planning.behavioral.evaluators.single_lane_action_spec_evaluator import \
    SingleLaneActionSpecEvaluator
from decision_making.src.planning.behavioral.evaluators.zero_value_approximator import ZeroValueApproximator
from decision_making.src.planning.behavioral.planner.single_step_behavioral_planner import SingleStepBehavioralPlanner
from decision_making.src.planning.route.route_planning_facade import RoutePlanningFacade
from decision_making.src.planning.route.binary_cost_based_route_planner import BinaryCostBasedRoutePlanner
from decision_making.src.planning.trajectory.trajectory_planning_facade import TrajectoryPlanningFacade
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.planning.trajectory.werling_planner import WerlingPlanner
from decision_making.src.prediction.ego_aware_prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.state.state_module import StateModule
import os
from rte.python.logger.AV_logger import AV_Logger
from rte.python.os import catch_interrupt_signals
from rte.python.parser import av_argument_parser

DEFAULT_MAP_FILE = Paths.get_repo_path() + '/../common_data/maps/PG_split.bin'


class DmInitialization:
    """
    This class contains the module initializations
    """

    @staticmethod
    def create_state_module() -> StateModule:
        logger = AV_Logger.get_logger(STATE_MODULE_NAME_FOR_LOGGING)

        pubsub = PubSub()
        state_module = StateModule(pubsub, logger, None)
        return state_module

    @staticmethod
    def create_route_planner() -> RoutePlanningFacade:
        logger = AV_Logger.get_logger(ROUTE_PLANNING_NAME_FOR_LOGGING)

        pubsub = PubSub()

        planner = BinaryCostBasedRoutePlanner()

        route_planning_module = RoutePlanningFacade(pubsub=pubsub, logger=logger, route_planner=planner)
        return route_planning_module

    @staticmethod
    def create_behavioral_planner() -> BehavioralPlanningFacade:
        logger = AV_Logger.get_logger(BEHAVIORAL_PLANNING_NAME_FOR_LOGGING)

        pubsub = PubSub()

        predictor = RoadFollowingPredictor(logger)

        action_space = ActionSpaceContainer(logger, [StaticActionSpace(logger, DEFAULT_STATIC_RECIPE_FILTERING),
                                                     DynamicActionSpace(logger, predictor,
                                                                        DEFAULT_DYNAMIC_RECIPE_FILTERING),
                                                     RoadSignActionSpace(logger, predictor,
                                                                         DEFAULT_DYNAMIC_RECIPE_FILTERING)],
                                            )

        recipe_evaluator = None
        action_spec_evaluator = SingleLaneActionSpecEvaluator(logger)  # RuleBasedActionSpecEvaluator(logger)
        value_approximator = ZeroValueApproximator(logger)

        action_spec_filtering = DEFAULT_ACTION_SPEC_FILTERING
        planner = SingleStepBehavioralPlanner(action_space, recipe_evaluator, action_spec_evaluator,
                                              action_spec_filtering, value_approximator, predictor, logger)

        behavioral_module = BehavioralPlanningFacade(pubsub=pubsub, logger=logger,
                                                     behavioral_planner=planner, last_trajectory=None)
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


def main():
    av_argument_parser.parse_arguments()
    # register termination signal handler
    logger = AV_Logger.get_logger(DM_MANAGER_NAME_FOR_LOGGING)
    logger.debug('%d: (DM main) registered signal handler', os.getpid())
    catch_interrupt_signals()

    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    modules_list = \
        [
            DmProcess(lambda: DmInitialization.create_route_planner(),
                      trigger_type=DmTriggerType.DM_TRIGGER_PERIODIC,
                      trigger_args={'period': ROUTE_PLANNING_MODULE_PERIOD},
                      name='RP'),

            DmProcess(lambda: DmInitialization.create_state_module(),
                      trigger_type=DmTriggerType.DM_TRIGGER_NONE,
                      trigger_args={},
                      name='SM'),

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


if __name__ == '__main__':
    main()
