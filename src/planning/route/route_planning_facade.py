from logging import Logger
import time
import traceback
import rte.python.profiler as prof
from interface.Rte_Types.python.uc_system import UC_SYSTEM_ROUTE_PLAN
from interface.Rte_Types.python.uc_system import UC_SYSTEM_SCENE_STATIC
from decision_making.src.exceptions import MsgDeserializationError, RoutePlanningException
from decision_making.src.global_constants import LOG_MSG_ROUTE_PLANNER_OUTPUT, \
    LOG_MSG_ROUTE_PLANNER_IMPL_TIME, ROUTE_PLANNING_NAME_FOR_METRICS, LOG_MSG_SCENE_STATIC_RECEIVED
from decision_making.src.infra.dm_module import DmModule
from decision_making.src.infra.pubsub import PubSub
from decision_making.src.messages.scene_common_messages import Header, Timestamp
from decision_making.src.messages.scene_static_message import SceneStatic
from decision_making.src.messages.route_plan_message import RoutePlan, DataRoutePlan
from decision_making.src.planning.route.binary_cost_based_route_planner import CostBasedRoutePlanner
from decision_making.src.planning.route.route_planner import RoutePlannerInputData
from decision_making.src.scene.scene_static_model import SceneStaticModel
from decision_making.src.utils.dm_profiler import DMProfiler
from decision_making.src.utils.metric_logger.metric_logger import MetricLogger


class RoutePlanningFacade(DmModule):
    def __init__(self, pubsub: PubSub, logger: Logger, route_planner: CostBasedRoutePlanner):
        """
        :param pubsub:
        :param logger:
        :param route_planner:
        """
        super().__init__(pubsub=pubsub, logger=logger)
        self.planner = route_planner
        self.timestamp = None
        self.logger.info("Initialized Route Planner Facade.")
        MetricLogger.init(ROUTE_PLANNING_NAME_FOR_METRICS)

    def _start_impl(self):
        """Subscribe to messages"""
        self.pubsub.subscribe(UC_SYSTEM_SCENE_STATIC, None)

    def _stop_impl(self):
        """Unsubscribe from messages"""
        self.pubsub.unsubscribe(UC_SYSTEM_SCENE_STATIC)

    def _periodic_action_impl(self) -> None:
        """
        The main function of the route planner. It read the most up-to-date scene static base, includning the navigation route and lane
        attributes, processes them into internal data structures (as described in RoutePlannerInputData() class).
        The main planner function then uses this intrernal data structure to come up with the route plan (as described in class
        DataRoutePlan() ) The results are then published to the behavior planner.
        """
        try:
            # Read inputs
            start_time = time.time()

            with DMProfiler(self.__class__.__name__ + '.get_scene_static'):
                scene_static = self._get_current_scene_static()

            SceneStaticModel.get_instance().set_scene_static(scene_static)

            route_planner_input = RoutePlannerInputData()
            route_planner_input.reformat_input_data(scene=scene_static.s_Data.s_SceneStaticBase,
                                                    nav_plan=scene_static.s_Data.s_NavigationPlan)

            # Plan
            with DMProfiler(self.__class__.__name__ + '.plan'):
                route_plan = self.__planner.plan(route_planner_input)

            # Write outputs
            self._publish_results(route_plan)

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

    @prof.ProfileFunction()
    def _get_current_scene_static(self) -> SceneStatic:
        is_success, serialized_scene_static = self.pubsub.get_latest_sample(topic=UC_SYSTEM_SCENE_STATIC)

        if serialized_scene_static is None:
            raise MsgDeserializationError("Pubsub message queue for %s topic is empty or topic isn\'t subscribed" %
                                          UC_SYSTEM_SCENE_STATIC)
        scene_static = SceneStatic.deserialize(serialized_scene_static)
        self.timestamp = Timestamp.from_seconds(scene_static.s_Header.s_Timestamp.timestamp_in_seconds)
        self.logger.debug('%s: %f' % (LOG_MSG_SCENE_STATIC_RECEIVED, scene_static.s_Header.s_Timestamp.timestamp_in_seconds))
        return scene_static

    @prof.ProfileFunction()
    def _publish_results(self, s_Data: DataRoutePlan) -> None:

        final_route_plan = RoutePlan(s_Header=Header(e_Cnt_SeqNum=0, s_Timestamp=self.timestamp, e_Cnt_version=0), s_Data = s_Data)

        self.pubsub.publish(UC_SYSTEM_ROUTE_PLAN, final_route_plan.serialize())
        self.logger.debug("{} {}".format(LOG_MSG_ROUTE_PLANNER_OUTPUT, final_route_plan))

    @property
    def planner(self):
        """Getter for planner property"""
        return self.__planner

    @planner.setter
    def planner(self, planner):
        """Setter for planner property"""
        self.__planner = planner
