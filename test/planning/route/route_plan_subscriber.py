import numpy as np
import traceback

from logging import Logger
from typing import List
from rte.python.logger.AV_logger import AV_Logger

from interface.Rte_Types.python.uc_system import UC_SYSTEM_ROUTE_PLAN
from interface.Rte_Types.python.uc_system import UC_SYSTEM_SCENE_STATIC
from interface.Rte_Types.python.uc_system import UC_SYSTEM_TAKEOVER

from decision_making.src.exceptions import MsgDeserializationError
from decision_making.src.infra.pubsub import PubSub
from decision_making.src.infra.dm_module import DmModule
from decision_making.src.global_constants import LOG_MSG_SCENE_STATIC_RECEIVED ,\
     BEHAVIORAL_PLANNING_NAME_FOR_LOGGING
from decision_making.src.messages.route_plan_message import RoutePlan
from decision_making.src.messages.scene_static_message import SceneStatic
from decision_making.src.messages.takeover_message import Takeover
from decision_making.test.planning.behavioral.mock_behavioral_facade import BehavioralFacadeMock
from decision_making.src.state.state import OccupancyState, State, ObjectSize, EgoState, DynamicObject
from decision_making.src.state.map_state import MapState


def generate_mock_state(ego_lane_id:int, ego_lane_station:float) -> State :

    # Ego state
    ego_vel = 10
    car_size = ObjectSize(length=2.5, width=1.5, height=1.0)

    map_state = MapState(np.array([ego_lane_station, ego_vel, 0, 0, 0, 0]), ego_lane_id)
    ego_state = EgoState.create_from_map_state(obj_id=0, timestamp=0, map_state=map_state, size=car_size, confidence=1)

    # Stub of occupancy grid
    occupancy_state = OccupancyState(0, np.array([]), np.array([]))

    dynamic_objects: List[DynamicObject] = list()

    return State(is_sampled=False, occupancy_state=occupancy_state, dynamic_objects=dynamic_objects, ego_state=ego_state)


class RoutePlanSubscriber(DmModule):
    def __init__(self, pubsub: PubSub, logger: Logger) -> None:
        """
        :param pubsub:
        :param logger:
        """
        super().__init__(pubsub=pubsub, logger=logger)
        self.logger.info("Initialized Behavioral Planner Facade.")

    def _start_impl(self):
        self.pubsub.subscribe(UC_SYSTEM_SCENE_STATIC, None)
        self.pubsub.subscribe(UC_SYSTEM_ROUTE_PLAN, None)

    def _stop_impl(self):
        self.pubsub.unsubscribe(UC_SYSTEM_SCENE_STATIC)
        self.pubsub.unsubscribe(UC_SYSTEM_ROUTE_PLAN)

    def _periodic_action_impl(self) -> None:
        """
        The main function of the behavioral planner. It read the most up-to-date state and navigation plan,
         processes them into the behavioral state, and then performs behavioral planning. The results are then published
          to the trajectory planner and as debug information to the visualizer.
        :return: void
        """

        try:
            # get mock state
            mock_state = generate_mock_state(ego_lane_id = 101, ego_lane_station = 0)

            # get the scene static data for MapUtils setup
            scene_static_data = self._get_current_scene_static()

            # get current route plan
            route_plan = self._get_current_route_plan()

            # calculate the takeover message
            behavior_facade_mock = BehavioralFacadeMock(pubsub=PubSub(), logger=AV_Logger.get_logger(BEHAVIORAL_PLANNING_NAME_FOR_LOGGING),
                                                        trajectory_params=None,
                                                        visualization_msg=None, trigger_pos=None)

            takeover_msg = behavior_facade_mock._mock_takeover_message(route_plan_data=route_plan.s_Data,
                                                                        ego_state=mock_state.ego_state,
                                                                        scene_static=scene_static_data)

            # publish takeover message
            self._publish_takeover(takeover_msg)

            # print values
            self._print_results(route_plan, takeover_msg)

        except Exception as e:
            self.logger.critical("UNHANDLED EXCEPTION IN BEHAVIORAL FACADE: %s. Trace: %s" %
                                 (e, traceback.format_exc()))

    def _get_current_scene_static(self) -> SceneStatic:
        is_success, serialized_scene_static = self.pubsub.get_latest_sample(topic=UC_SYSTEM_SCENE_STATIC)
        # TODO Move the raising of the exception to LCM code. Do the same in trajectory facade
        if serialized_scene_static is None:
            raise MsgDeserializationError("Pubsub message queue for %s topic is empty or topic isn\'t subscribed" %
                                          UC_SYSTEM_SCENE_STATIC)
        scene_static = SceneStatic.deserialize(serialized_scene_static)
        self.logger.debug("%s: %f" % (LOG_MSG_SCENE_STATIC_RECEIVED, scene_static.s_Header.s_Timestamp.timestamp_in_seconds))
        return scene_static

    def _get_current_route_plan(self) -> RoutePlan:
        """
        Returns the last received route plan data
        We assume that if no updates have been received since the last call,
        then we will output the last received state.
        :return: deserialized RoutePlan
        """
        is_success, input_route_plan = self.pubsub.get_latest_sample(topic=UC_SYSTEM_ROUTE_PLAN)
        if input_route_plan is None:
            raise MsgDeserializationError("Pubsub message queue for %s topic is empty or topic isn\'t subscribed" %
                                          UC_SYSTEM_ROUTE_PLAN)
        object_route_plan = RoutePlan.deserialize(input_route_plan)
        self.logger.debug("Received route plan: %s" % object_route_plan)
        return object_route_plan

    def _publish_takeover(self, takeover_msg:Takeover) -> None :
        self.pubsub.publish(UC_SYSTEM_TAKEOVER, takeover_msg.serialize())

    def _print_results(self, route_plan:RoutePlan, takeover_msg:Takeover) :

        print("------------   ROUTE MESSAGE BEGIN  ---------------------")
        for i in range(route_plan.s_Data.e_Cnt_num_road_segments):
            print("ROAD SEGMENT: " , route_plan.s_Data.a_i_road_segment_ids[i], "\n")
            for j in range(route_plan.s_Data.a_Cnt_num_lane_segments[i]):
                print("Lane Segment ID: ", route_plan.s_Data.as_route_plan_lane_segments[i][j].e_i_lane_segment_id ,  \
                      "   Lane End Cost:  ", route_plan.s_Data.as_route_plan_lane_segments[i][j].e_cst_lane_end_cost , \
                      "   Lane Occupancy Cost:  ", route_plan.s_Data.as_route_plan_lane_segments[i][j].e_cst_lane_occupancy_cost , "\n" )

        print("TAKEOVER FLAG:  ", takeover_msg.s_Data.e_b_is_takeover_needed , "\n")
        print("------------   ROUTE MESSAGE END  ---------------------","\n\n")