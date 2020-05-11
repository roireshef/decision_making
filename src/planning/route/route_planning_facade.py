import time
import traceback
from logging import Logger

import numpy as np
import rte.python.profiler as prof
from decision_making.src.exceptions import MissingMapInformation, raises
from decision_making.src.exceptions import MsgDeserializationError, RoutePlanningException
from decision_making.src.global_constants import LOG_MSG_ROUTE_PLANNER_OUTPUT, \
    LOG_MSG_ROUTE_PLANNER_IMPL_TIME, ROUTE_PLANNING_NAME_FOR_METRICS, LOG_MSG_SCENE_STATIC_RECEIVED
from decision_making.src.infra.dm_module import DmModule
from decision_making.src.infra.pubsub import PubSub
from decision_making.src.messages.route_plan_message import RoutePlan, DataRoutePlan
from decision_making.src.messages.scene_common_messages import Header, Timestamp
from decision_making.src.messages.scene_static_message import SceneStatic, SceneStaticBase, NavigationPlan
from decision_making.src.planning.route.binary_cost_based_route_planner import RoutePlanner
from decision_making.src.planning.route.route_planner_input_data import RoutePlannerInputData
from decision_making.src.scene.scene_static_model import SceneStaticModel
from decision_making.src.utils.dm_profiler import DMProfiler
from decision_making.src.utils.metric_logger.metric_logger import MetricLogger
from interface.Rte_Types.python.uc_system.uc_system_scene_static import UC_SYSTEM_SCENE_STATIC
from interface.Rte_Types.python.uc_system.uc_system_route_plan import UC_SYSTEM_ROUTE_PLAN


class RoutePlanningFacade(DmModule):
    def __init__(self, pubsub: PubSub, logger: Logger, route_planner: RoutePlanner):
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

            self._check_scene_data_validity(scene=scene_static.s_Data.s_SceneStaticBase,
                                            nav_plan=scene_static.s_Data.s_NavigationPlan)

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
            scene_static_road_segments = scene_static.s_Data.s_SceneStaticBase.as_scene_road_segment or []
            scene_static_road_segment_ids = [rs.e_i_road_segment_id for rs in scene_static_road_segments]
            inputs_str_dict = {"navigation plan road segments": scene_static.s_Data.s_NavigationPlan.a_i_road_segment_ids.tolist(),
                               "scene static base road segments": scene_static_road_segment_ids}
            self.logger.error("RoutePlanningException: %s. When inputs are %s. Trace: %s", e, inputs_str_dict,
                              traceback.format_exc())

        except Exception as e:
            self.logger.critical("RoutePlanningFacade: UNHANDLED EXCEPTION: %s. Trace: %s",
                                 e, traceback.format_exc())

    @prof.ProfileFunction()
    def _get_current_scene_static(self) -> SceneStatic:
        is_success, serialized_scene_static = self.pubsub.get_latest_sample(topic=UC_SYSTEM_SCENE_STATIC)

        if serialized_scene_static is None or not serialized_scene_static.s_Data.e_b_Valid:
            raise MsgDeserializationError("Pubsub message queue for %s topic is empty or topic isn\'t subscribed" %
                                          UC_SYSTEM_SCENE_STATIC)
        scene_static = SceneStatic.deserialize(serialized_scene_static)
        self.timestamp = Timestamp.from_seconds(scene_static.s_Header.s_Timestamp.timestamp_in_seconds)
        self.logger.debug('%s: %f' % (LOG_MSG_SCENE_STATIC_RECEIVED, scene_static.s_Header.s_Timestamp.timestamp_in_seconds))
        return scene_static

    @staticmethod   # Made static method especially as this method doesn't access the classes states/variables
    @raises(MissingMapInformation)
    def _check_scene_data_validity(scene: SceneStaticBase, nav_plan: NavigationPlan) -> None:
        if not scene.as_scene_lane_segments:
            raise MissingMapInformation("RoutePlanner MissingInputInformation: Empty scene.as_scene_lane_segments")

        if not scene.as_scene_road_segment:
            raise MissingMapInformation("RoutePlanner MissingInputInformation: Empty scene.as_scene_road_segment")

        if not nav_plan.a_i_road_segment_ids.size:  # np.ndarray type
            raise MissingMapInformation("RoutePlanner MissingInputInformation: Empty NAV Plan")

        scene_road_segment_ids = [road.e_i_road_segment_id for road in scene.as_scene_road_segment]
        is_nav_road_segment_in_scene_static = np.in1d(nav_plan.a_i_road_segment_ids, scene_road_segment_ids)
        if not is_nav_road_segment_in_scene_static.all():
            raise MissingMapInformation("RoutePlanner MissingInputInformation: road segments %s are in the navigation "
                                        "plan but not in scene static list of road segments" %
                                        nav_plan.a_i_road_segment_ids[~is_nav_road_segment_in_scene_static])

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
