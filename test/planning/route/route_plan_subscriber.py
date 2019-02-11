import time
import traceback
from logging import Logger

import numpy as np

from common_data.interface.Rte_Types.python import Rte_Types_pubsub as pubsub_topics
from decision_making.src.infra.pubsub import PubSub
from decision_making.src.global_constants import DISTANCE_TO_SET_TAKEOVER_FLAG
from decision_making.src.infra.dm_module import DmModule
from decision_making.src.planning.types import CartesianExtendedState
# from decision_making.src.planning.utils.localization_utils import LocalizationUtils
# from decision_making.src.state.state import State
from decision_making.src.utils.metric_logger import MetricLogger

# from decision_making.src.utils.map_utils import MapUtils
from decision_making.src.messages.route_plan_message import RoutePlan
from decision_making.src.messages.takeover_message import Takeover, DataTakeover
from decision_making.src.messages.scene_common_messages import Header, Timestamp
from decision_making.src.planning.types import C_Y, FS_SX

class StateMock() :
    def __init__(self, ego_lane_seg_id, ego_road_seg_id, ego_lane_seg_station, ego_lane_length) -> None :
        self.ego_lane_seg_id = ego_lane_seg_id
        self.ego_road_seg_id = ego_road_seg_id
        self.ego_lane_seg_station = ego_lane_seg_station
        self.ego_lane_length = ego_lane_length


class RoutePlanSubscriber(DmModule):
    def __init__(self, pubsub: PubSub, logger: Logger) -> None:
        """
        :param pubsub:
        :param logger:
        """
        super().__init__(pubsub=pubsub, logger=logger)
        self.logger.info("Initialized Behavioral Planner Facade.")

    def _start_impl(self):
        #self.pubsub.subscribe(pubsub_topics.PubSubMessageTypes["UC_SYSTEM_STATE_LCM"], None)
        self.pubsub.subscribe(pubsub_topics.PubSubMessageTypes["UC_SYSTEM_ROUTE_PLAN"],None)

    # TODO: unsubscribe once logic is fixed in LCM
    def _stop_impl(self):
        pass

    def _periodic_action_impl(self) -> None:
        """
        The main function of the behavioral planner. It read the most up-to-date state and navigation plan,
         processes them into the behavioral state, and then performs behavioral planning. The results are then published
          to the trajectory planner and as debug information to the visualizer.
        :return: void
        """

        try:
            start_time = time.time()
            #state = self._get_current_state()
            state_mock = StateMock(1,1,0,100)

            # get current route plan 
            route_plan = self._get_current_route_plan()
            # calculate the takeover message
            #takeover_msg = self._set_takeover_message(route_plan , state)
            takeover_msg_mock = self._set_takeover_message_mock(route_plan , state_mock)
            
            # publish takeover message
            self._publish_takeover(takeover_msg_mock)

            # print values
            self._print_results(route_plan, takeover_msg_mock)

        except Exception as e:
            self.logger.critical("UNHANDLED EXCEPTION IN BEHAVIORAL FACADE: %s. Trace: %s" %
                                 (e, traceback.format_exc()))

    # def _get_current_state(self) -> State:
    #     is_success, input_state = self.pubsub.get_latest_sample(topic=pubsub_topics.PubSubMessageTypes["UC_SYSTEM_STATE_LCM"], timeout=1)
    #     # TODO Move the raising of the exception to LCM code. Do the same in trajectory facade
    #     object_state = State.deserialize(input_state)
    #     return object_state


    def _get_current_route_plan(self) -> RoutePlan:
        is_success, input_route_plan = self.pubsub.get_latest_sample(topic=pubsub_topics.PubSubMessageTypes["UC_SYSTEM_ROUTE_PLAN"], timeout=1)
        object_route_plan = RoutePlan.deserialize(input_route_plan)
        self.logger.debug("Received route plan: %s" % object_route_plan)
        return object_route_plan

    # def _set_takeover_message(self, route_plan:RoutePlan, state:State ) -> Takeover:
        
    #     # find current lane segment ID
    #     ego_lane_id = MapUtils.get_closest_lane(state.ego_state.cartesian_state[:(C_Y+1)])
    #     # find current road segment ID
    #     curr_road_segment_id = MapUtils.get_road_segment_id_from_lane_id(ego_lane_id)
    #     # find road segment index in route plan 2-d array 
    #     route_plan_idx = [i for i in range(route_plan.s_Data.e_Cnt_num_road_segments) \
    #                         if route_plan.s_Data.a_i_road_segment_ids[i]==curr_road_segment_id ]
        
    #     assert(len(route_plan_idx)==1 and route_plan_idx[0] >= 0 and route_plan_idx[0] < route_plan.s_Data.e_Cnt_num_road_segments )

    #     row_idx = route_plan_idx[0]

    #      # find station on the current lane 
    #     ego_station = state.ego_state.map_state[FS_SX]
    #     # find length of the lane segment
    #     ego_lane_length = MapUtils.get_lane_length(ego_lane_id)
    #     # distance to the end of current road (lane) segment
    #     dist_to_end = ego_lane_length - ego_station 

    #     # check the end costs for the current road segment lanes
    #     blockage_flag = True
    #     for i in range(row_idx,route_plan.s_Data.e_Cnt_num_road_segments):
            
    #         for j in range(route_plan.s_Data.a_Cnt_num_lane_segments[i]) :
    #             if route_plan.s_Data.as_route_plan_lane_segments[i][j].e_cst_lane_end_cost < 1 :
    #                 blockage_flag = False
    #                 break

    #         # check how many road segments are within the horizon
    #         if i > row_idx:
    #             next_road_lane_id = route_plan.s_Data.as_route_plan_lane_segments[i][0].e_i_lane_segment_id
    #             lane_length = MapUtils.get_lane_length(next_road_lane_id)
    #             dist_to_end += lane_length
    #         if dist_to_end >= DISTANCE_TO_SET_TAKEOVER_FLAG :
    #             break
        
    #     if blockage_flag == True and dist_to_end < DISTANCE_TO_SET_TAKEOVER_FLAG:
    #         takeover_flag = True

    #     # TODO check this timestamp
    #     timestamp_object = Timestamp.from_seconds(state.ego_state.timestamp_in_sec)

    #     takeover_msg = Takeover(s_Header=Header(e_Cnt_SeqNum=0, s_Timestamp=timestamp_object,e_Cnt_version=0) , \
    #                             s_Data = DataTakeover(takeover_flag) ) 

    #     return takeover_msg


    def _publish_takeover(self, takeover_msg:Takeover) -> None :
        self.pubsub.publish(pubsub_topics.PubSubMessageTypes["UC_SYSTEM_TAKEOVER"], takeover_msg.serialize())


    def _print_results(self, route_plan:RoutePlan, takeover_msg:Takeover) :
        
        for i in range(route_plan.s_Data.e_Cnt_num_road_segments):
            print("ROAD SEGMENT: " , route_plan.s_Data.a_i_road_segment_ids[i], "\n")
            for j in range(route_plan.s_Data.a_Cnt_num_lane_segments[i]):
                print("Lane Segment ID: ", route_plan.s_Data.as_route_plan_lane_segments[i][j].e_i_lane_segment_id ,  \
                      "   Lane End Cost:  ", route_plan.s_Data.as_route_plan_lane_segments[i][j].e_cst_lane_end_cost , \
                      "   Lane Occupancy Cost:  ", route_plan.s_Data.as_route_plan_lane_segments[i][j].e_cst_lane_occupancy_cost , "\n" )

        print("TAKEOVER FLAG:  ", takeover_msg.s_Data.e_b_is_takeover_needed , "\n")


    def _set_takeover_message_mock(self, route_plan:RoutePlan, state_mock:StateMock ) -> Takeover:
        
        # find current lane segment ID
        ego_lane_id = state_mock.ego_lane_seg_id
        # find current road segment ID
        curr_road_segment_id = state_mock.ego_road_seg_id
        # find road segment index in route plan 2-d array 
        route_plan_idx = [i for i in range(route_plan.s_Data.e_Cnt_num_road_segments) \
                            if route_plan.s_Data.a_i_road_segment_ids[i]==curr_road_segment_id ]
        
        assert(len(route_plan_idx)==1 and route_plan_idx[0] >= 0 and route_plan_idx[0] < route_plan.s_Data.e_Cnt_num_road_segments )

        row_idx = route_plan_idx[0]

         # find station on the current lane 
        ego_station = state_mock.ego_lane_seg_station
        # find length of the lane segment
        ego_lane_length = state_mock.ego_lane_length
        # distance to the end of current road (lane) segment
        dist_to_end = ego_lane_length - ego_station 

        # check the end costs for the current road segment lanes
        blockage_flag = True
        for i in range(row_idx,route_plan.s_Data.e_Cnt_num_road_segments):
            
            for j in range(route_plan.s_Data.a_Cnt_num_lane_segments[i]) :
                if route_plan.s_Data.as_route_plan_lane_segments[i][j].e_cst_lane_end_cost < 1 :
                    blockage_flag = False
                    break

            # check how many road segments are within the horizon
            #if i > row_idx:
            #    next_road_lane_id = route_plan.s_Data.as_route_plan_lane_segments[i][0].e_i_lane_segment_id
            #    lane_length = MapUtils.get_lane_length(next_road_lane_id)
            #    dist_to_end += lane_length
            #if dist_to_end >= DISTANCE_TO_SET_TAKEOVER_FLAG :
            #    break
        
        takeover_flag = False

        if blockage_flag == True and dist_to_end < DISTANCE_TO_SET_TAKEOVER_FLAG:
            takeover_flag = True

        # TODO check this timestamp
        #timestamp_object = Timestamp.from_seconds(state.ego_state.timestamp_in_sec)

        timestamp_object = Timestamp(0,0)

        takeover_msg = Takeover(s_Header=Header(e_Cnt_SeqNum=0, s_Timestamp=timestamp_object,e_Cnt_version=0) , \
                                s_Data = DataTakeover(takeover_flag) ) 

        return takeover_msg


    