from logging import Logger
from common_data.src.communication.pubsub.pubsub import PubSub
from common_data.interface.py.pubsub import Rte_Types_pubsub_topics as pubsub_topics
import time
import traceback
from logging import Logger

from decision_making.src.exceptions import MsgDeserializationError
from decision_making.src.infra.dm_module import DmModule
from decision_making.src.planning.route.route_planner import RoutePlanner
from decision_making.src.utils.metric_logger import MetricLogger
from decision_making.src.global_constants import LOG_MSG_ROUTE_PLANNER_OUTPUT, LOG_MSG_RECEIVED_STATE, \
    LOG_MSG_ROUTE_PLANNER_IMPL_TIME, ROUTE_PLANNING_NAME_FOR_METRICS, LOG_MSG_SCENE_STATIC_RECEIVED
    
from decision_making.src.messages.scene_static_lite_message import SceneStaticLite,DataSceneStaticLite
from decision_making.src.messages.route_plan_message import RoutePlan,RoutePlanLaneSegment, DataRoutePlan
from cost_based_route_planner import RoutePlannerInputData
    

class RoutePlanningFacade(DmModule):


    def __init__(self, pubsub: PubSub, logger: Logger, route_planner: RoutePlanner):
        """
        :param pubsub:
        :param logger:
        :param route_planner:
        """
        super().__init__(pubsub=pubsub, logger=logger)
        self.__planner = route_planner
        self.logger.info("Initialized Route Planner Facade.")
        MetricLogger.init(ROUTE_PLANNING_NAME_FOR_METRICS)

    def _start_impl(self):
        """Add comments"""
        self.pubsub.subscribe(pubsub_topics.SCENE_STATIC, None)
        pass

    def _stop_impl(self):
        """Add comments"""
        pass

    def _periodic_action_impl(self):
        """Add comments"""
        try:
            # Read inputs
            start_time = time.time()
            scene_static = self._get_current_scene_static()
            MainRoutePlanInputData = RoutePlannerInputData(scene_static)


            # Plan
            route_plan = self.__planner.plan(MainRoutePlanInputData)

            # Write outputs

            # Send plan to behavior
            self._publish_results(route_plan)

            # Send visualization data
            #self._publish_visualization(behavioral_visualization_message)

            self.logger.info("{} {}".format(LOG_MSG_ROUTE_PLANNER_IMPL_TIME, time.time() - start_time))

            MetricLogger.get_logger().report()

        except MsgDeserializationError as e:
            self.logger.warning("RoutePlanningFacade: MsgDeserializationError was raised. Skipping planning. " +
                                "Turn on debug logging level for more details. Trace: %s", traceback.format_exc())
            self.logger.debug(str(e))

        except Exception as e:
            self.logger.critical("RoutePlanningFacade: UNHANDLED EXCEPTION: %s. Trace: %s",
                                 e, traceback.format_exc())

    def _get_current_scene_static(self) -> SceneStaticLite:
        is_success, serialized_scene_static = self.pubsub.get_latest_sample(topic=pubsub_topics.SCENE_STATIC, timeout=1)
        # TODO Move the raising of the exception to LCM code. Do the same in trajectory facade
        if serialized_scene_static is None:
            raise MsgDeserializationError('Pubsub message queue for %s topic is empty or topic isn\'t subscribed',
                                          pubsub_topics.SCENE_STATIC)
        scene_static = SceneStaticLite.deserialize(serialized_scene_static)
        self.logger.debug('%s: %f' % (LOG_MSG_SCENE_STATIC_RECEIVED, scene_static.s_Header.s_Timestamp.timestamp_in_seconds))
        return scene_static
    
    def _publish_results(self, route_plan: RoutePlan) -> None:
        self.pubsub.publish(pubsub_topics.ROUTE_PLAN, route_plan.serialize())
        self.logger.debug("{} {}".format(LOG_MSG_ROUTE_PLANNER_OUTPUT, route_plan))

    @property
    def planner(self):
        """Add comments"""
        return self._planner

    @planner.setter
    def planner(self, planner):
        """Add comments"""
        self.__planner = planner
