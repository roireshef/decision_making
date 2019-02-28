import numpy as np
from typing import List

from common_data.interface.Rte_Types.python.sub_structures import (
    TsSYSRoutePlan,
    TsSYSDataRoutePlan,
    TsSYSRoutePlanLaneSegment)

from decision_making.src.messages.scene_common_messages import Header
from decision_making.src.global_constants import PUBSUB_MSG_IMPL

class RoutePlanLaneSegment(PUBSUB_MSG_IMPL):
    """
    Route Plan Lane Segment Information

    Args:            
        e_i_lane_segment_id: TODO: Add Comment
        e_cst_lane_occupancy_cost: TODO: Add Comment
        e_cst_lane_end_cost: TODO: Add Comment
    """
    e_i_lane_segment_id = int
    e_cst_lane_occupancy_cost = float
    e_cst_lane_end_cost = float

    def __init__(self, e_i_lane_segment_id: int, e_cst_lane_occupancy_cost: float, e_cst_lane_end_cost: float):
        self.e_i_lane_segment_id = e_i_lane_segment_id
        self.e_cst_lane_occupancy_cost = e_cst_lane_occupancy_cost
        self.e_cst_lane_end_cost = e_cst_lane_end_cost

    def serialize(self) -> TsSYSRoutePlanLaneSegment:
        pubsub_msg = TsSYSRoutePlanLaneSegment()
        
        # =====================================================================
        # This block of code was added due to a TypeError previously raised by the middleware.
        # At the time, the type of this variable was numpy.uint64, but the middleware was expecting an int.

        # If a numpy type, convert to built-in type
        if type(self.e_i_lane_segment_id).__module__ is np.__name__:
            self.e_i_lane_segment_id = self.e_i_lane_segment_id.item()
        
        # If still not type int, cast to int and print message
        if type(self.e_i_lane_segment_id) is not int:
            self.e_i_lane_segment_id = int(self.e_i_lane_segment_id)
            print("RoutePlanLaneSegment: e_i_lane_segment_id is not int")
        # =====================================================================

        pubsub_msg.e_i_lane_segment_id = self.e_i_lane_segment_id
        pubsub_msg.e_cst_lane_occupancy_cost = self.e_cst_lane_occupancy_cost
        pubsub_msg.e_cst_lane_end_cost = self.e_cst_lane_end_cost

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSRoutePlanLaneSegment):
        return cls(pubsubMsg.e_i_lane_segment_id,
                   pubsubMsg.e_cst_lane_occupancy_cost,
                   pubsubMsg.e_cst_lane_end_cost)

class DataRoutePlan(PUBSUB_MSG_IMPL):
    """
    Route Plan Output Data

    Args:
        e_b_is_valid: TODO: Add Comment
        e_Cnt_num_road_segments: TODO: Add Comment
        a_i_road_segment_ids: TODO: Add Comment
        a_Cnt_num_lane_segments: TODO: Add Comment
        as_route_plan_lane_segments: TODO: Add Comment
    """
    e_b_is_valid = bool
    e_Cnt_num_road_segments = int
    a_i_road_segment_ids = np.ndarray
    a_Cnt_num_lane_segments = np.ndarray
    as_route_plan_lane_segments = List[List[RoutePlanLaneSegment]]

    def __init__(self, e_b_is_valid: bool, e_Cnt_num_road_segments: int, a_i_road_segment_ids: np.ndarray,
                 a_Cnt_num_lane_segments: np.ndarray, as_route_plan_lane_segments: List[List[RoutePlanLaneSegment]]):
        self.e_b_is_valid = e_b_is_valid
        self.e_Cnt_num_road_segments = e_Cnt_num_road_segments
        self.a_i_road_segment_ids = a_i_road_segment_ids
        self.a_Cnt_num_lane_segments = a_Cnt_num_lane_segments
        self.as_route_plan_lane_segments = as_route_plan_lane_segments

    def serialize(self) -> TsSYSDataRoutePlan:
        pubsub_msg = TsSYSDataRoutePlan()

        pubsub_msg.e_b_is_valid = self.e_b_is_valid
        pubsub_msg.e_Cnt_num_road_segments = self.e_Cnt_num_road_segments
        pubsub_msg.a_i_road_segment_ids = self.a_i_road_segment_ids
        pubsub_msg.a_Cnt_num_lane_segments = self.a_Cnt_num_lane_segments
        
        for i in range(pubsub_msg.e_Cnt_num_road_segments):
            for j in range(pubsub_msg.a_Cnt_num_lane_segments[i]):
                pubsub_msg.as_route_plan_lane_segments[i][j] = self.as_route_plan_lane_segments[i][j].serialize()

        return pubsub_msg
    
    

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSDataRoutePlan):
        as_route_plan_lane_segments = [[RoutePlanLaneSegment.deserialize(pubsubMsg.as_route_plan_lane_segments[i][j]) \
                                        for j in range(pubsubMsg.a_Cnt_num_lane_segments[i])] \
                                       for i in range(pubsubMsg.e_Cnt_num_road_segments)]

        return cls(pubsubMsg.e_b_is_valid,
                   pubsubMsg.e_Cnt_num_road_segments,
                   pubsubMsg.a_i_road_segment_ids[:pubsubMsg.e_Cnt_num_road_segments],
                   pubsubMsg.a_Cnt_num_lane_segments[:pubsubMsg.e_Cnt_num_road_segments],
                   as_route_plan_lane_segments)

    def __str__(self):
        print_route = "\n"
        for i, road_segment in enumerate(self.as_route_plan_lane_segments):
            a_i_lane_segment_ids = []
            a_cst_lane_occupancy_costs = []
            a_cst_lane_end_costs = []
            for j, lane_segment in enumerate(road_segment):
                a_i_lane_segment_ids.append(lane_segment.e_i_lane_segment_id)
                a_cst_lane_occupancy_costs.append(lane_segment.e_cst_lane_occupancy_cost)
                a_cst_lane_end_costs.append(lane_segment.e_cst_lane_end_cost)
            print_route = print_route + "lane_segment_ids "+str(a_i_lane_segment_ids)+"\n"
            print_route = print_route + "lane_occupancy_costs "+str(a_cst_lane_occupancy_costs)+"\n"
            print_route = print_route + "lane_end_costs "+str(a_cst_lane_end_costs)+"\n"
            print_route = print_route +"\n"

        return print_route


class RoutePlan(PUBSUB_MSG_IMPL):
    """
    Class that represents the ROUTE_PLAN topic
    
    Args:
        s_Header: TODO: Add Comment
        s_Data: TODO: Add Comment
    """
    s_Header = Header
    s_Data = DataRoutePlan

    def __init__(self, s_Header: Header, s_Data: DataRoutePlan):
        self.s_Header = s_Header
        self.s_Data = s_Data

    def serialize(self) -> TsSYSRoutePlan:
        pubsub_msg = TsSYSRoutePlan()

        pubsub_msg.s_Header = self.s_Header.serialize()
        pubsub_msg.s_Data = self.s_Data.serialize()
        
        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSRoutePlan):
        return cls(Header.deserialize(pubsubMsg.s_Header),
                   DataRoutePlan.deserialize(pubsubMsg.s_Data))
