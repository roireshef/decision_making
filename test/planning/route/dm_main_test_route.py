from logging import Logger
from os import getpid

import numpy as np
import os

from decision_making.paths import Paths

from decision_making.src.infra.pubsub import PubSub
from common_data.interface.Rte_Types.python.Rte_Types_pubsub import PubSubMessageTypes
from decision_making.src.global_constants import STATE_MODULE_NAME_FOR_LOGGING, \
    NAVIGATION_PLANNING_NAME_FOR_LOGGING, \
    ROUTE_PLANNING_NAME_FOR_LOGGING, \
    BEHAVIORAL_PLANNING_NAME_FOR_LOGGING, \
    TRAJECTORY_PLANNING_NAME_FOR_LOGGING, \
    DM_MANAGER_NAME_FOR_LOGGING, BEHAVIORAL_PLANNING_MODULE_PERIOD, TRAJECTORY_PLANNING_MODULE_PERIOD, ROUTE_PLANNING_MODULE_PERIOD
from decision_making.src.manager.dm_manager import DmManager
from decision_making.src.manager.dm_process import DmProcess
from decision_making.src.manager.dm_trigger import DmTriggerType
from decision_making.src.planning.route.route_planning_facade import RoutePlanningFacade
from decision_making.src.planning.route.cost_based_route_planner import CostBasedRoutePlanner
# from decision_making.src.state.state_module import StateModule
from mapping.src.service.map_service import MapService
from rte.python.logger.AV_logger import AV_Logger
from rte.python.os import catch_interrupt_signals
from decision_making.test.planning.route.route_plan_subscriber import RoutePlanSubscriber
from decision_making.test.planning.route.scene_static_publisher import SceneStaticPublisher

DEFAULT_MAP_FILE = Paths.get_repo_path() + '/../common_data/maps/PG_split.bin'

class DmInitialization:
    """
    This class contains the module initializations
    """

    # @staticmethod
    # def create_state_module(map_file: str=DEFAULT_MAP_FILE) -> StateModule:
    #     logger = AV_Logger.get_logger(STATE_MODULE_NAME_FOR_LOGGING)

    #     pubsub = PubSub()
    #     # MapService should be initialized in each process according to the given map_file
    #     MapService.initialize(map_file)
    #     state_module = StateModule(pubsub, logger, None)
    #     return state_module

    @staticmethod
    def create_route_planner(map_file:str=DEFAULT_MAP_FILE):
        logger = AV_Logger.get_logger(ROUTE_PLANNING_NAME_FOR_LOGGING)

        pubsub = PubSub()
        # MapService should be initialized in each process according to the given map_file
        MapService.initialize(map_file)

        planner = CostBasedRoutePlanner()

        route_planning_module = RoutePlanningFacade(pubsub=pubsub, logger=logger, route_planner=planner)
        return route_planning_module

    @staticmethod
    def create_scene_static_publisher(map_file: str=DEFAULT_MAP_FILE) -> SceneStaticPublisher:
       logger = AV_Logger.get_logger("SCENE_STATIC_PUBLISHER")

       pubsub = PubSub()
       # MapService should be initialized in each process according to the given map_file
       MapService.initialize(map_file)

       scene_static_publisher_module = SceneStaticPublisher(pubsub=pubsub, logger=logger)
       return scene_static_publisher_module

    @staticmethod
    def create_route_subscriber(map_file: str=DEFAULT_MAP_FILE) -> RoutePlanSubscriber:
        logger = AV_Logger.get_logger("ROUTE_PLAN_SUBSCRIBER")

        pubsub = PubSub()
        # MapService should be initialized in each process according to the given map_file
        MapService.initialize(map_file)

        route_subscriber_module = RoutePlanSubscriber(pubsub=pubsub, logger=logger)
        return route_subscriber_module

def main():
    # register termination signal handler
    logger = AV_Logger.get_logger(DM_MANAGER_NAME_FOR_LOGGING)
    logger.debug('%d: (DM main) registered signal handler', getpid())
    catch_interrupt_signals()

    modules_list = \
        [
            DmProcess(lambda:DmInitialization.create_scene_static_publisher(DEFAULT_MAP_FILE),
                     trigger_type=DmTriggerType.DM_TRIGGER_PERIODIC,
                     trigger_args={'period': ROUTE_PLANNING_MODULE_PERIOD}),

            DmProcess(lambda: DmInitialization.create_route_planner(DEFAULT_MAP_FILE),
                      trigger_type=DmTriggerType.DM_TRIGGER_PERIODIC,
                      trigger_args={'period': ROUTE_PLANNING_MODULE_PERIOD}),

            DmProcess(lambda:DmInitialization.create_route_subscriber(DEFAULT_MAP_FILE),
                      trigger_type=DmTriggerType.DM_TRIGGER_PERIODIC,
                      trigger_args={'period': ROUTE_PLANNING_MODULE_PERIOD})
        ]

    manager = DmManager(logger, modules_list)
    manager.start_modules()
    try:
        manager.wait_for_submodules()
    except KeyboardInterrupt:
        logger.info('%d: (DM main) interrupted by signal', getpid())
        pass
    finally:
        manager.stop_modules()

if __name__ == '__main__':
    main()
