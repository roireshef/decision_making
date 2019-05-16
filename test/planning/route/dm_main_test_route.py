from os import getpid
from decision_making.paths import Paths
from decision_making.src.infra.pubsub import PubSub
from decision_making.src.global_constants import ROUTE_PLANNING_NAME_FOR_LOGGING, \
    DM_MANAGER_NAME_FOR_LOGGING, ROUTE_PLANNING_MODULE_PERIOD
from decision_making.src.manager.dm_manager import DmManager
from decision_making.src.manager.dm_process import DmProcess
from decision_making.src.manager.dm_trigger import DmTriggerType
from decision_making.src.planning.route.route_planning_facade import RoutePlanningFacade
from decision_making.src.planning.route.binary_cost_based_route_planner import BinaryCostBasedRoutePlanner
from decision_making.test.mapping.service.map_service import MapService
from rte.python.logger.AV_logger import AV_Logger
from rte.python.os import catch_interrupt_signals
from decision_making.test.planning.route.route_plan_subscriber import RoutePlanSubscriber
from decision_making.test.planning.route.scene_static_publisher import SceneStaticPublisher
from decision_making.test.planning.route.scene_static_publisher_facade import SceneStaticPublisherFacade

DEFAULT_MAP_FILE = Paths.get_repo_path() + '/../common_data/maps/PG_split.bin'

class DmInitialization:
    """
    This class contains the module initializations
    """
    @staticmethod
    def create_route_planner(map_file:str=DEFAULT_MAP_FILE):
        logger = AV_Logger.get_logger(ROUTE_PLANNING_NAME_FOR_LOGGING)

        pubsub = PubSub()
        # MapService should be initialized in each process according to the given map_file
        MapService.initialize(map_file)

        planner = BinaryCostBasedRoutePlanner()

        return RoutePlanningFacade(pubsub=pubsub, logger=logger, route_planner=planner)

    @staticmethod
    def create_scene_static_publisher(map_file: str=DEFAULT_MAP_FILE) -> SceneStaticPublisherFacade:
        logger = AV_Logger.get_logger("SCENE_STATIC_PUBLISHER")

        pubsub = PubSub()
        # MapService should be initialized in each process according to the given map_file
        MapService.initialize(map_file)

        # Initialize Publisher
        road_segment_ids = [1, 2]

        lane_segment_ids = [[101, 102],
                            [201, 202]]

        navigation_plan = [1, 2]

        scene_static_publisher = SceneStaticPublisher(road_segment_ids=road_segment_ids,
                                                      lane_segment_ids=lane_segment_ids,
                                                      navigation_plan=navigation_plan)

        return SceneStaticPublisherFacade(pubsub=pubsub, logger=logger, publisher=scene_static_publisher)

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
                     trigger_args={'period': ROUTE_PLANNING_MODULE_PERIOD}, name='SP'),

            DmProcess(lambda: DmInitialization.create_route_planner(DEFAULT_MAP_FILE),
                      trigger_type=DmTriggerType.DM_TRIGGER_PERIODIC,
                      trigger_args={'period': ROUTE_PLANNING_MODULE_PERIOD}, name='RP'),

            DmProcess(lambda:DmInitialization.create_route_subscriber(DEFAULT_MAP_FILE),
                      trigger_type=DmTriggerType.DM_TRIGGER_PERIODIC,
                      trigger_args={'period': ROUTE_PLANNING_MODULE_PERIOD}, name='RS')
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
