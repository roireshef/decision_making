import numpy as np
from typing import List, Dict, Tuple

from interface.Rte_Types.python.sub_structures.TsSYS_RoutePlan import TsSYSRoutePlan
from interface.Rte_Types.python.sub_structures.TsSYS_DataRoutePlan import TsSYSDataRoutePlan
from interface.Rte_Types.python.sub_structures.TsSYS_RoutePlanLaneSegment import TsSYSRoutePlanLaneSegment
from decision_making.src.exceptions import RoadNotFound, raises
from decision_making.src.messages.serialization import PUBSUB_MSG_IMPL
from decision_making.src.messages.scene_common_messages import Header, Timestamp
from decision_making.src.planning.types import LaneSegmentID, LaneOccupancyCost, LaneEndCost


class RoutePlanLaneSegment(PUBSUB_MSG_IMPL):
    e_i_lane_segment_id = int
    e_cst_lane_occupancy_cost = float
    e_cst_lane_end_cost = float

    def __init__(self, e_i_lane_segment_id: int, e_cst_lane_occupancy_cost: float, e_cst_lane_end_cost: float):
        """
        Route Plan Lane Segment Information
        :param e_i_lane_segment_id: Lane segment ID
        :param e_cst_lane_occupancy_cost: Cost of being within the lane while driving through the associated road segment
        :param e_cst_lane_end_cost: Cost of being within the lane at the end of the associated road segment. In other words, this cost
            deals with transitions between road segments.
        """
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

    def __str__(self):
        print_route_plan_lane_segment = "lane_segment_id " + str(self.e_i_lane_segment_id) + \
                                        " lane_occupancy_cost " + str(self.e_cst_lane_occupancy_cost) + \
                                        " lane_end_cost " + str(self.e_cst_lane_end_cost)
        return print_route_plan_lane_segment

RoutePlanRoadSegment = List[RoutePlanLaneSegment]   # RoutePlanLaneSegment : struct -> Contains Route plan end and occupancy costs,
                                                    # along with the lane identifier to convey routing relevant information in a structure.
                                                    # RoutePlanRoadSegment: List[RoutePlanLaneSegment] are List of RoutePlanLaneSegment(s).
                                                    # Contains all the RoutePlanLaneSegment in a RoadSegment.

RoutePlanRoadSegments = List[RoutePlanRoadSegment]  # RoutePlanRoadSegments:List[RoutePlanRoadSegment] are
                                                    # List of (List of RoutePlanLaneSegment(s)).
                                                    # Contains all the RoutePlanLaneSegment in a Road.


class DataRoutePlan(PUBSUB_MSG_IMPL):
    e_b_is_valid = bool
    e_Cnt_num_road_segments = int
    a_i_road_segment_ids = np.ndarray
    a_Cnt_num_lane_segments = np.ndarray
    as_route_plan_lane_segments = RoutePlanRoadSegments
    s_data_creation_time = Timestamp

    def __init__(self,
                 e_b_is_valid: bool,
                 e_Cnt_num_road_segments: int,
                 a_i_road_segment_ids: np.ndarray,
                 a_Cnt_num_lane_segments: np.ndarray,
                 as_route_plan_lane_segments: RoutePlanRoadSegments,
                 s_data_creation_time: Timestamp):
        """
        Route Plan Output Data
        :param e_b_is_valid: Set to true when the data is valid
        :param e_Cnt_num_road_segments: Number of road segments in the route plan
        :param a_i_road_segment_ids: Ordered array of road segment IDs
        :param a_Cnt_num_lane_segments: Array containing the number of lane segments in each road segment
        :param as_route_plan_lane_segments: 2D array containing lane segment information
        :param s_data_creation_time:
        """
        self.e_b_is_valid = e_b_is_valid
        self.e_Cnt_num_road_segments = e_Cnt_num_road_segments
        self.a_i_road_segment_ids = a_i_road_segment_ids
        self.a_Cnt_num_lane_segments = a_Cnt_num_lane_segments
        self.as_route_plan_lane_segments = as_route_plan_lane_segments
        self.s_DataCreationTime = s_data_creation_time

    def serialize(self) -> TsSYSDataRoutePlan:
        pubsub_msg = TsSYSDataRoutePlan()

        pubsub_msg.e_b_is_valid = self.e_b_is_valid
        pubsub_msg.e_Cnt_num_road_segments = self.e_Cnt_num_road_segments
        pubsub_msg.a_i_road_segment_ids = self.a_i_road_segment_ids
        pubsub_msg.a_Cnt_num_lane_segments = self.a_Cnt_num_lane_segments

        for i in range(pubsub_msg.e_Cnt_num_road_segments):
            for j in range(pubsub_msg.a_Cnt_num_lane_segments[i]):
                pubsub_msg.as_route_plan_lane_segments[i][j] = self.as_route_plan_lane_segments[i][j].serialize()

        pubsub_msg.s_DataCreationTime = self.s_DataCreationTime.serialize()

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSDataRoutePlan):
        as_route_plan_lane_segments = [[RoutePlanLaneSegment.deserialize(pubsubMsg.as_route_plan_lane_segments[i][j])
                                        for j in range(pubsubMsg.a_Cnt_num_lane_segments[i])]
                                       for i in range(pubsubMsg.e_Cnt_num_road_segments)]

        return cls(
            e_b_is_valid=pubsubMsg.e_b_is_valid,
            e_Cnt_num_road_segments=pubsubMsg.e_Cnt_num_road_segments,
            a_i_road_segment_ids=pubsubMsg.a_i_road_segment_ids[:pubsubMsg.e_Cnt_num_road_segments],
            a_Cnt_num_lane_segments=pubsubMsg.a_Cnt_num_lane_segments[:pubsubMsg.e_Cnt_num_road_segments],
            as_route_plan_lane_segments=as_route_plan_lane_segments,
            s_data_creation_time=Timestamp.deserialize(pubsubMsg.s_DataCreationTime),
        )

    def __str__(self)->str:
        for i, road_segment in enumerate(self.as_route_plan_lane_segments):
            a_i_lane_segment_ids = []
            a_cst_lane_occupancy_costs = []
            a_cst_lane_end_costs = []
            for j, lane_segment in enumerate(road_segment):
                a_i_lane_segment_ids.append(lane_segment.e_i_lane_segment_id)
                a_cst_lane_occupancy_costs.append(lane_segment.e_cst_lane_occupancy_cost)
                a_cst_lane_end_costs.append(lane_segment.e_cst_lane_end_cost)
            print_route = "lane_segment_ids: " + str(a_i_lane_segment_ids) + \
                          "lane_occupancy_costs: " + str(a_cst_lane_occupancy_costs) + \
                          "lane_end_costs " + str(a_cst_lane_end_costs) + "\n"

        return print_route


class RoutePlan(PUBSUB_MSG_IMPL):
    s_Header = Header
    s_Data = DataRoutePlan

    def __init__(self, s_Header: Header, s_Data: DataRoutePlan):
        """
        Class that represents the ROUTE_PLAN topic
        :param s_Header: General Information
        :param s_Data: Message Data
        """
        self.s_Header = s_Header
        self.s_Data = s_Data

    def to_costs_dict(self) -> Dict[LaneSegmentID, Tuple[LaneOccupancyCost, LaneEndCost]]:
        """
         returns a complete dictionary of lane costs:
         keys are lane_segment_ids and values are tuples containing the lane costs: (occupancy cost, end cost). The following are two
         constants that hold the indices and can be used to access each cost: LANE_OCCUPANCY_COST_IND = 0 and LANE_END_COST_IND = 1
        :return:
        """
        # TODO: cache when route plan header data is accurate
        return {lane_segment.e_i_lane_segment_id: (lane_segment.e_cst_lane_occupancy_cost, lane_segment.e_cst_lane_end_cost)
                for road_segment in self.s_Data.as_route_plan_lane_segments
                for lane_segment in road_segment}

    @raises(RoadNotFound)
    def get_road_index_in_plan(self, road_id, start=None, end=None):
        """
        Given a road_id, returns the index of this road_id in the navigation plan (represented as list of roads)
        :param road_id: the request road_id to look for in the plan
        :param start: optional. starting index to look from in the plan (inclusive)
        :param end: optional. ending index to look up to in the plan (inclusive)
        :return: index of road_id in the plan
        """
        road_ids = self.s_Data.a_i_road_segment_ids
        try:
            if start is None:
                start = 0
            if end is None:
                end = len(road_ids)
            return np.where(road_ids[start:(end+1)] == road_id)[0][0] + start
        except IndexError:
            raise RoadNotFound("Road ID {} is not in clipped (indices: [{}, {}]) plan's road-IDs [{}]"
                               .format(road_id, start, end, road_ids[start:(end+1)]))

    def serialize(self) -> TsSYSRoutePlan:
        pubsub_msg = TsSYSRoutePlan()

        pubsub_msg.s_Header = self.s_Header.serialize()
        pubsub_msg.s_Data = self.s_Data.serialize()

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSRoutePlan):
        return cls(Header.deserialize(pubsubMsg.s_Header),
                   DataRoutePlan.deserialize(pubsubMsg.s_Data))
