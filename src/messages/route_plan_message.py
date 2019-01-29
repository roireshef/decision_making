import numpy as np
from typing import List

from common_data.interface.py.idl_generated_files.Rte_Types.TsSYS_RoutePlan import TsSYSRoutePlan
from common_data.interface.py.idl_generated_files.Rte_Types.sub_structures.TsSYS_DataRoutePlan import TsSYSDataRoutePlan
from common_data.interface.py.idl_generated_files.Rte_Types.sub_structures.TsSYS_RoutePlanLaneSegment import TsSYSRoutePlanLaneSegment

from decision_making.src.messages.scene_common_messages import Timestamp, Header
from decision_making.src.global_constants import PUBSUB_MSG_IMPL

class RoutePlanLaneSegment(PUBSUB_MSG_IMPL):
    e_i_lane_segment_id = int
    e_cst_lane_occupancy_cost = float
    e_cst_lane_end_cost = float

    def __init__(self, e_i_lane_segment_id: int, e_cst_lane_occupancy_cost: float, e_cst_lane_end_cost: float):
        """
        Route Plan Lane Segment Information

         param e_i_lane_segment_id: TODO: Add Comment
         param e_cst_lane_occupancy_cost: TODO: Add Comment
         param e_cst_lane_end_cost: TODO: Add Comment
        """
        self.e_i_lane_segment_id = e_i_lane_segment_id
        self.e_cst_lane_occupancy_cost = e_cst_lane_occupancy_cost
        self.e_cst_lane_end_cost = e_cst_lane_end_cost

    def serialize(self) -> TsSYSDataRoutePlan:
        pubsub_msg = TsSYSDataRoutePlan()

        pubsub_msg.e_i_lane_segment_id = self.e_i_lane_segment_id
        pubsub_msg.e_cst_lane_occupancy_cost = self.e_cst_lane_occupancy_cost
        pubsub_msg.e_cst_lane_end_cost = self.e_cst_lane_end_cost

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSDataRoutePlan):
        return cls(pubsubMsg.e_b_Valid,
                   pubsubMsg.e_l_perception_horizon_front,
                   pubsubMsg.e_l_perception_horizon_rear)

class DataRoutePlan(PUBSUB_MSG_IMPL):
    e_Cnt_num_road_segments = int
    a_i_road_segment_ids = np.ndarray
    a_Cnt_num_lane_segments = np.ndarray
    as_route_plan_lane_segments = List[List[RoutePlanLaneSegment]]

    def __init__(self, e_Cnt_num_road_segments: int, a_i_road_segment_ids: np.ndarray, a_Cnt_num_lane_segments: np.ndarray,
                 as_route_plan_lane_segments: List[List[RoutePlanLaneSegment]]):
        """
        Route Planner Output

         param e_Cnt_num_road_segments: TODO: Add Comment
         param a_i_road_segment_ids: TODO: Add Comment
         param a_Cnt_num_lane_segments: TODO: Add Comment
         param as_route_plan_lane_segments: TODO: Add Comment
        """
        self.e_Cnt_num_road_segments = e_Cnt_num_road_segments
        self.a_i_road_segment_ids = a_i_road_segment_ids
        self.a_Cnt_num_lane_segments = a_Cnt_num_lane_segments
        self.as_route_plan_lane_segments = as_route_plan_lane_segments

    def serialize(self) -> TsSYSDataRoutePlan:
        pubsub_msg = TsSYSDataRoutePlan()

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

        return cls(pubsubMsg.e_Cnt_num_road_segments,
                   pubsubMsg.a_i_road_segment_ids[:pubsubMsg.e_Cnt_num_road_segments],
                   pubsubMsg.a_Cnt_num_lane_segments[:pubsubMsg.e_Cnt_num_road_segments],
                   as_route_plan_lane_segments)

class RoutePlan(PUBSUB_MSG_IMPL):
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
