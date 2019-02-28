from logging import Logger
from decision_making.src.infra.pubsub import PubSub
from common_data.interface.Rte_Types.python import Rte_Types_pubsub as pubsub_topics
import time
import traceback

from decision_making.src.exceptions import MsgDeserializationError, RoutePlanningException
from decision_making.src.infra.dm_module import DmModule
from decision_making.src.planning.route.route_planner import RoutePlanner, RoutePlannerInputData
from decision_making.src.utils.metric_logger import MetricLogger
from decision_making.src.global_constants import LOG_MSG_ROUTE_PLANNER_OUTPUT, LOG_MSG_RECEIVED_STATE, \
    LOG_MSG_ROUTE_PLANNER_IMPL_TIME, ROUTE_PLANNING_NAME_FOR_METRICS, LOG_MSG_SCENE_STATIC_RECEIVED
from decision_making.src.messages.scene_common_messages import Header, Timestamp, MapOrigin
from decision_making.src.messages.scene_static_message import SceneStatic, SceneStaticBase, NavigationPlan
from decision_making.src.messages.route_plan_message import RoutePlan, RoutePlanLaneSegment, DataRoutePlan


class RoutePlanningFacade(DmModule):


    def __init__(self, pubsub: PubSub, logger: Logger, route_planner: RoutePlanner):
        """
        :param pubsub:
        :param logger:
        :param route_planner:
        """
        super().__init__(pubsub=pubsub, logger=logger)
        self.planner = route_planner
        self.logger.info("Initialized Route Planner Facade.")
        MetricLogger.init(ROUTE_PLANNING_NAME_FOR_METRICS)

    def _start_impl(self):
        """Subscribe to messages"""
        self.pubsub.subscribe(pubsub_topics.PubSubMessageTypes["UC_SYSTEM_SCENE_STATIC"], None)

    def _stop_impl(self):
        """Unsubscribe from messages"""
        self.pubsub.unsubscribe(pubsub_topics.PubSubMessageTypes["UC_SYSTEM_SCENE_STATIC"])

    def _periodic_action_impl(self)->None:
        """
        The main function of the route planner. It read the most up-to-date scene static base, includning the navigation route and lane
        attributes, processes them into internal data structures (as described in RoutePlannerInputData() class).
        The main planner function then uses this intrernal data structure to come up with the route plan (as described in class
        DataRoutePlan() ) The results are then published to the behavior planner.
        """

        try:
            # Read inputs
            start_time = time.time()
            ss_base, ss_nav = self._get_current_scene_static()
            route_planner_input = RoutePlannerInputData()
            route_planner_input.reformat_input_data(scene=ss_base, nav_plan=ss_nav)

            # Plan
            route_plan = self.__planner.plan(route_planner_input)

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

        except RoutePlanningException as e:
            self.logger.warning(e)

        except Exception as e:
            self.logger.critical("RoutePlanningFacade: UNHANDLED EXCEPTION: %s. Trace: %s",
                                 e, traceback.format_exc())

    def _get_current_scene_static(self) -> (SceneStaticBase, NavigationPlan):
        is_success, serialized_scene_static = self.pubsub.get_latest_sample(
            topic=pubsub_topics.PubSubMessageTypes["UC_SYSTEM_SCENE_STATIC"], timeout=1)
        # TODO Move the raising of the exception to LCM code. Do the same in trajectory facade
        if serialized_scene_static is None:
            raise MsgDeserializationError('Pubsub message queue for %s topic is empty or topic isn\'t subscribed',
                                          pubsub_topics.PubSubMessageTypes["UC_SYSTEM_SCENE_STATIC"])
        scene_static = SceneStatic.deserialize(serialized_scene_static)
        self.logger.debug('%s: %f' % (LOG_MSG_SCENE_STATIC_RECEIVED, scene_static.s_Header.s_Timestamp.timestamp_in_seconds))
        return scene_static.s_Data.s_SceneStaticBase, scene_static.s_Data.s_NavigationPlan

    def _publish_results(self, s_Data: DataRoutePlan) -> None:
        timestamp_object = Timestamp.from_seconds(0)

        final_route_plan = RoutePlan(s_Header=Header(e_Cnt_SeqNum=0, s_Timestamp=timestamp_object,e_Cnt_version=0), s_Data = s_Data)

        self.pubsub.publish(pubsub_topics.PubSubMessageTypes["UC_SYSTEM_ROUTE_PLAN"], final_route_plan.serialize())
        self.logger.debug("{} {}".format(LOG_MSG_ROUTE_PLANNER_OUTPUT, final_route_plan))

    @property
    def planner(self):
        """Getter for planner property"""
        return self.__planner

    @planner.setter
    def planner(self, planner):
        """Setter for planner property"""
        self.__planner = planner
