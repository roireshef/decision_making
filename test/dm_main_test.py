from os import getpid

import numpy as np

from common_data.lcm.config import config_defs
from common_data.lcm.python.Communication.lcmpubsub import LcmPubSub
from common_data.src.communication.pubsub.pubsub_factory import create_pubsub
from decision_making.src import global_constants
from decision_making.src.dm_main import DmInitialization
from decision_making.src.global_constants import BEHAVIORAL_PLANNING_MODULE_PERIOD, TRAJECTORY_PLANNING_MODULE_PERIOD, \
    DM_MANAGER_NAME_FOR_LOGGING, TRAJECTORY_PLANNING_NAME_FOR_LOGGING, TRAJECTORY_TIME_RESOLUTION, \
    BEHAVIORAL_PLANNING_NAME_FOR_LOGGING, EGO_LENGTH, EGO_WIDTH, EGO_HEIGHT, PREDICTION_LOOKAHEAD_COMPENSATION_RATIO
from decision_making.src.manager.dm_manager import DmManager
from decision_making.src.manager.dm_process import DmProcess
from decision_making.src.manager.dm_trigger import DmTriggerType
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.messages.trajectory_parameters import TrajectoryParams
from decision_making.src.messages.visualization.behavioral_visualization_message import BehavioralVisualizationMsg
from decision_making.src.planning.behavioral.behavioral_facade import BehavioralFacade
from decision_making.src.planning.behavioral.policies.semantic_actions_grid_policy import SemanticActionsGridPolicy
from decision_making.src.planning.trajectory.trajectory_planning_facade import TrajectoryPlanningFacade
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.planning.types import C_Y, C_X
from decision_making.src.prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.state.state import ObjectSize
from decision_making.test import constants
from decision_making.test.constants import TP_MOCK_FIXED_TRAJECTORY_FILENAME, BP_MOCK_FIXED_SPECS
from decision_making.test.planning.behavioral.mock_behavioral_facade import BehavioralFacadeMock
from decision_making.test.planning.trajectory.fixed_trajectory_planner import FixedTrajectoryPlanner
from decision_making.test.utils import Utils
from mapping.src.service.map_service import MapService
from rte.python.logger.AV_logger import AV_Logger
from rte.python.os import catch_interrupt_signals


class DmMockInitialization:
    @staticmethod
    def create_trajectory_planner() -> TrajectoryPlanningFacade:
        logger = AV_Logger.get_logger(TRAJECTORY_PLANNING_NAME_FOR_LOGGING)
        pubsub = create_pubsub(config_defs.LCM_SOCKET_CONFIG, LcmPubSub)

        # Init map
        MapService.initialize()
        predictor = RoadFollowingPredictor(logger)

        fixed_trajectory = Utils.read_trajectory(TP_MOCK_FIXED_TRAJECTORY_FILENAME)

        step_size = TRAJECTORY_PLANNING_MODULE_PERIOD / TRAJECTORY_TIME_RESOLUTION
        planner = FixedTrajectoryPlanner(logger, predictor, fixed_trajectory, step_size,
                                         fixed_trajectory[0, :(C_Y+1)])

        strategy_handlers = {TrajectoryPlanningStrategy.HIGHWAY: planner,
                             TrajectoryPlanningStrategy.PARKING: planner,
                             TrajectoryPlanningStrategy.TRAFFIC_JAM: planner}

        trajectory_planning_module = TrajectoryPlanningFacade(pubsub=pubsub, logger=logger,
                                                              strategy_handlers=strategy_handlers)
        return trajectory_planning_module

    @staticmethod
    def create_behavioral_planner() -> BehavioralFacade:
        logger = AV_Logger.get_logger(BEHAVIORAL_PLANNING_NAME_FOR_LOGGING)
        pubsub = create_pubsub(config_defs.LCM_SOCKET_CONFIG, LcmPubSub)

        # Init map
        MapService.initialize()

        road = MapService.get_instance().get_road(BP_MOCK_FIXED_SPECS['ROAD_ID'])
        desired_lat = road.lane_width * (BP_MOCK_FIXED_SPECS['LANE_NUM'] + 0.5)
        nav_plan = NavigationPlanMsg(road_ids=np.array([BP_MOCK_FIXED_SPECS['ROAD_ID']]))
        trigger_point = BP_MOCK_FIXED_SPECS['TRIGGER_POINT']

        # get the trigger-point's longitude from the road's beginning
        _, init_point_longitude, _, _, _ = MapService.get_instance().convert_global_to_road_coordinates(
            trigger_point[0], trigger_point[1], 0)

        ref_route, _ = MapService.get_instance().get_lookahead_points(
            initial_road_id=BP_MOCK_FIXED_SPECS['ROAD_ID'],
            initial_lon=init_point_longitude,
            lookahead_dist=BP_MOCK_FIXED_SPECS['LOOKAHEAD_DISTANCE'] * PREDICTION_LOOKAHEAD_COMPENSATION_RATIO,
            desired_lat=desired_lat,
            navigation_plan=nav_plan
        )

        _, target_lon = MapService.get_instance().advance_on_plan(
            initial_road_id=BP_MOCK_FIXED_SPECS['ROAD_ID'],
            initial_lon=init_point_longitude,
            lookahead_dist=BP_MOCK_FIXED_SPECS['LOOKAHEAD_DISTANCE'],
            navigation_plan=nav_plan
        )

        target_pose, target_yaw = MapService.get_instance().convert_road_to_global_coordinates(
            road_id=BP_MOCK_FIXED_SPECS['ROAD_ID'], lon=target_lon, lat=desired_lat)

        target_state = np.append(target_pose[[C_X, C_Y]], [target_yaw, BP_MOCK_FIXED_SPECS['TARGET_VELOCITY']])

        cost_params = SemanticActionsGridPolicy._generate_cost_params(
            road_id=BP_MOCK_FIXED_SPECS['ROAD_ID'], ego_size=ObjectSize(EGO_LENGTH, EGO_WIDTH, EGO_HEIGHT),
            reference_route_latitude=desired_lat)

        # time here is relative for the moment, but upon triggering of the BehavioralFacadeMock,
        # the ego.timestamp_in_sec is being added.
        params = TrajectoryParams(strategy=TrajectoryPlanningStrategy.HIGHWAY,
                                  reference_route=ref_route,
                                  target_state=target_state,
                                  cost_params=cost_params,
                                  time=BP_MOCK_FIXED_SPECS['PLANNING_TIME'])

        viz_msg = BehavioralVisualizationMsg(reference_route=ref_route)

        behavioral_module = BehavioralFacadeMock(pubsub=pubsub, logger=logger,
                                                 trigger_pos=BP_MOCK_FIXED_SPECS['TRIGGER_POINT'],
                                                 trajectory_params=params, visualization_msg=viz_msg)
        return behavioral_module


def main():
    """
    initializes DM planning pipeline. for switching between BP/TP impl./mock make sure to comment out the relevant
    instantiation in modules_list.
    """
    modules_list = \
        [
            DmProcess(DmInitialization.create_navigation_planner,
                      trigger_type=DmTriggerType.DM_TRIGGER_PERIODIC,
                      trigger_args={'period': BEHAVIORAL_PLANNING_MODULE_PERIOD}),

            DmProcess(DmInitialization.create_state_module,
                      trigger_type=DmTriggerType.DM_TRIGGER_NONE,
                      trigger_args={}),

            # DmProcess(DmInitialization.create_behavioral_planner,
            DmProcess(DmMockInitialization.create_behavioral_planner,
                      trigger_type=DmTriggerType.DM_TRIGGER_PERIODIC,
                      trigger_args={'period': BEHAVIORAL_PLANNING_MODULE_PERIOD}),

            DmProcess(DmInitialization.create_trajectory_planner,
            # DmProcess(DmMockInitialization.create_trajectory_planner,
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
