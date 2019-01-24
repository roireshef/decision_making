from enum import Enum
from typing import List

import numpy as np

from common_data.interface.py.idl_generated_files.Rte_Types.sub_structures import TsSYSAdjacentLane, TsSYSBoundaryPoint, TsSYSLaneCoupling, \
    TsSYSStaticTrafficFlowControl, TsSYSDynamicStatus, TsSYSDynamicTrafficFlowControl, \
    TsSYSSceneLaneSegment, TsSYSLaneSegmentConnectivity
from common_data.interface.py.idl_generated_files.Rte_Types.TsSYS_SceneStatic import TsSYSSceneStatic
from common_data.interface.py.idl_generated_files.Rte_Types.sub_structures.TsSYS_DataSceneStatic import \
    TsSYSDataSceneStatic
from common_data.interface.py.idl_generated_files.Rte_Types.sub_structures.TsSYS_SceneRoadIntersection import \
    TsSYSSceneRoadIntersection
from common_data.interface.py.idl_generated_files.Rte_Types.sub_structures.TsSYS_SceneRoadSegment import \
    TsSYSSceneRoadSegment
from decision_making.src.global_constants import PUBSUB_MSG_IMPL
from decision_making.src.messages.scene_common_messages import Timestamp, MapOrigin, Header
from rte.python.logger.AV_logger import AV_Logger


from decision_making.src.messages.scene_static_message import MapLaneType,MapRoadSegmentType,MovingDirection, ManeuverType, MapLaneMarkerType, \
     RoadObjectType, TrafficSignalState, SceneRoadSegment, SceneRoadIntersection, AdjacentLane, LaneSegmentConnectivity,LaneCoupling,StaticTrafficFlowControl,\
     DynamicStatus,DynamicTrafficFlowControl

MAX_NOMINAL_PATH_POINT_FIELDS = 10




class LaneAttributes(PUBSUB_MSG_IMPL):
    e_Cnt_num_active_lane_attributes = int 
    e_i_active_lane_attribute_indices = List[int]
    e_cmp_lane_attributes = List[int]
    e_cmp_lane_attribute_confidences = List[float]

    def __init__(self, e_e_status: TrafficSignalState, e_Pct_confidence: float):
        """
        Status of Dynamic traffic-flow-control device, eg. red-yellow-green (not relevant for M0)
        :param e_e_status:
        :param e_Pct_confidence:
        """
        self.e_e_status = e_e_status
        self.e_Pct_confidence = e_Pct_confidence

    def serialize(self) -> TsSYSDynamicStatus:
        pubsub_msg = TsSYSDynamicStatus()

        pubsub_msg.e_e_status = self.e_e_status.value
        pubsub_msg.e_Pct_confidence = self.e_Pct_confidence

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSDynamicStatus):
        return cls(TrafficSignalState(pubsubMsg.e_e_status), pubsubMsg.e_Pct_confidence)

class SceneLaneSegmentLite(PUBSUB_MSG_IMPL):
    e_i_lane_segment_id = int
    e_i_road_segment_id = int
    e_e_lane_type = MapLaneType
    e_Cnt_static_traffic_flow_control_count = int
    as_static_traffic_flow_control = List[StaticTrafficFlowControl]
    e_Cnt_dynamic_traffic_flow_control_count = int
    as_dynamic_traffic_flow_control = List[DynamicTrafficFlowControl]
    e_Cnt_left_adjacent_lane_count = int
    as_left_adjacent_lanes = List[AdjacentLane]
    e_Cnt_right_adjacent_lane_count = int
    as_right_adjacent_lanes = List[AdjacentLane]
    e_Cnt_downstream_lane_count = int
    as_downstream_lanes = List[LaneSegmentConnectivity]
    e_Cnt_upstream_lane_count = int
    as_upstream_lanes = List[LaneSegmentConnectivity]
    e_v_nominal_speed = float
    e_i_downstream_road_intersection_id = int
    e_Cnt_lane_coupling_count = int
    as_lane_coupling = List[LaneCoupling]

    def __init__(self, e_i_lane_segment_id: int, e_i_road_segment_id: int, e_e_lane_type: MapLaneType,
                 e_Cnt_static_traffic_flow_control_count: int,
                 as_static_traffic_flow_control: List[StaticTrafficFlowControl],
                 e_Cnt_dynamic_traffic_flow_control_count: int,
                 as_dynamic_traffic_flow_control: List[DynamicTrafficFlowControl],
                 e_Cnt_left_adjacent_lane_count: int, as_left_adjacent_lanes: List[AdjacentLane],
                 e_Cnt_right_adjacent_lane_count: int, as_right_adjacent_lanes: List[AdjacentLane],
                 e_Cnt_downstream_lane_count: int, as_downstream_lanes: List[LaneSegmentConnectivity],
                 e_Cnt_upstream_lane_count: int, as_upstream_lanes: List[LaneSegmentConnectivity],
                 e_v_nominal_speed: float,
                 e_i_downstream_road_intersection_id: int, e_Cnt_lane_coupling_count: int,
                 as_lane_coupling: List[LaneCoupling]):
        """
        Lane-segment information
        :param e_i_lane_segment_id: ID of this lane-segment
        :param e_i_road_segment_id: ID of the road-segment that this lane-segment belongs to
        :param e_e_lane_type: Type of lane-segment
        :param e_Cnt_static_traffic_flow_control_count: Total number of static traffic-flow-control devices in this lane-segment (not relevant for M0)
        :param as_static_traffic_flow_control: Static traffic-flow-control devices in this lane-segment (not relevant for M0)
        :param e_Cnt_dynamic_traffic_flow_control_count: Total number of dynamic traffic-flow-control devices in this lane-segment (not relevant for M0)
        :param as_dynamic_traffic_flow_control: Dynamic traffic-flow-control devices in this lane-segment (not relevant for M0)
        :param e_Cnt_left_adjacent_lane_count: Total number of lane-segments to the left of this lane-segment
        :param as_left_adjacent_lanes: Lane-segments to the left of this lane-segment
        :param e_Cnt_right_adjacent_lane_count: Total number of lane-segments to the right of this lane-segment
        :param as_right_adjacent_lanes: Lane-segments to the right of this lane-segment
        :param e_Cnt_downstream_lane_count: Total number of lane-segments downstream of this lane-segment
        :param as_downstream_lanes: Lane-segments downstream of this lane-segment
        :param e_Cnt_upstream_lane_count: Total number of lane-segments upstream of this lane-segment
        :param as_upstream_lanes: Lane-segments upstream of this lane-segment
        :param e_v_nominal_speed: Nominal speed (i.e. speed limit) of this lane-segment
        :param e_i_downstream_road_intersection_id: ID of the Road-Intersection that is immediately downstream from this lane-segment (0 if not applicable)
        :param e_Cnt_lane_coupling_count: Total number of lane-couplings for this lane-segment
        :param as_lane_coupling: Lane-couplings for this lane-segment
        """
        self.e_i_lane_segment_id = e_i_lane_segment_id
        self.e_i_road_segment_id = e_i_road_segment_id
        self.e_e_lane_type = e_e_lane_type
        self.e_Cnt_static_traffic_flow_control_count = e_Cnt_static_traffic_flow_control_count
        self.as_static_traffic_flow_control = as_static_traffic_flow_control
        self.e_Cnt_dynamic_traffic_flow_control_count = e_Cnt_dynamic_traffic_flow_control_count
        self.as_dynamic_traffic_flow_control = as_dynamic_traffic_flow_control
        self.e_Cnt_left_adjacent_lane_count = e_Cnt_left_adjacent_lane_count
        self.as_left_adjacent_lanes = as_left_adjacent_lanes
        self.e_Cnt_right_adjacent_lane_count = e_Cnt_right_adjacent_lane_count
        self.as_right_adjacent_lanes = as_right_adjacent_lanes
        self.e_Cnt_downstream_lane_count = e_Cnt_downstream_lane_count
        self.as_downstream_lanes = as_downstream_lanes
        self.e_Cnt_upstream_lane_count = e_Cnt_upstream_lane_count
        self.as_upstream_lanes = as_upstream_lanes
        self.e_v_nominal_speed = e_v_nominal_speed
        self.e_i_downstream_road_intersection_id = e_i_downstream_road_intersection_id
        self.e_Cnt_lane_coupling_count = e_Cnt_lane_coupling_count
        self.as_lane_coupling = as_lane_coupling

    def serialize(self) -> TsSYSSceneLaneSegment:
        pubsub_msg = TsSYSSceneLaneSegment()

        pubsub_msg.e_i_lane_segment_id = self.e_i_lane_segment_id
        pubsub_msg.e_i_road_segment_id = self.e_i_road_segment_id
        pubsub_msg.e_e_lane_type = self.e_e_lane_type.value

        pubsub_msg.e_Cnt_static_traffic_flow_control_count = self.e_Cnt_static_traffic_flow_control_count
        for i in range(pubsub_msg.e_Cnt_static_traffic_flow_control_count):
            pubsub_msg.as_static_traffic_flow_control[i] = self.as_static_traffic_flow_control[i].serialize()

        pubsub_msg.e_Cnt_dynamic_traffic_flow_control_count = self.e_Cnt_dynamic_traffic_flow_control_count
        for i in range(pubsub_msg.e_Cnt_dynamic_traffic_flow_control_count):
            pubsub_msg.as_dynamic_traffic_flow_control[i] = self.as_dynamic_traffic_flow_control[i].serialize()

        pubsub_msg.e_Cnt_left_adjacent_lane_count = self.e_Cnt_left_adjacent_lane_count
        for i in range(pubsub_msg.e_Cnt_left_adjacent_lane_count):
            pubsub_msg.as_left_adjacent_lanes[i] = self.as_left_adjacent_lanes[i].serialize()

        pubsub_msg.e_Cnt_right_adjacent_lane_count = self.e_Cnt_right_adjacent_lane_count
        for i in range(pubsub_msg.e_Cnt_right_adjacent_lane_count):
            pubsub_msg.as_right_adjacent_lanes[i] = self.as_right_adjacent_lanes[i].serialize()

        pubsub_msg.e_Cnt_downstream_lane_count = self.e_Cnt_downstream_lane_count
        for i in range(pubsub_msg.e_Cnt_downstream_lane_count):
            pubsub_msg.as_downstream_lanes[i] = self.as_downstream_lanes[i].serialize()

        pubsub_msg.e_Cnt_upstream_lane_count = self.e_Cnt_upstream_lane_count
        for i in range(pubsub_msg.e_Cnt_upstream_lane_count):
            pubsub_msg.as_upstream_lanes[i] = self.as_upstream_lanes[i].serialize()

        pubsub_msg.e_v_nominal_speed = self.e_v_nominal_speed

        pubsub_msg.e_i_downstream_road_intersection_id = self.e_i_downstream_road_intersection_id

        pubsub_msg.e_Cnt_lane_coupling_count = self.e_Cnt_lane_coupling_count
        for i in range(pubsub_msg.e_Cnt_lane_coupling_count):
            pubsub_msg.as_lane_coupling[i] = self.as_lane_coupling[i].serialize()

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSSceneLaneSegment):
        as_static_traffic_flow_control = list()
        for i in range(pubsubMsg.e_Cnt_static_traffic_flow_control_count):
            as_static_traffic_flow_control.append(
                StaticTrafficFlowControl.deserialize(pubsubMsg.as_static_traffic_flow_control[i]))

        as_dynamic_traffic_flow_control = list()
        for i in range(pubsubMsg.e_Cnt_dynamic_traffic_flow_control_count):
            as_dynamic_traffic_flow_control.append(
                DynamicTrafficFlowControl.deserialize(pubsubMsg.as_dynamic_traffic_flow_control[i]))

        as_left_adjacent_lanes = list()
        for i in range(pubsubMsg.e_Cnt_left_adjacent_lane_count):
            as_left_adjacent_lanes.append(AdjacentLane.deserialize(pubsubMsg.as_left_adjacent_lanes[i]))

        as_right_adjacent_lanes = list()
        for i in range(pubsubMsg.e_Cnt_right_adjacent_lane_count):
            as_right_adjacent_lanes.append(AdjacentLane.deserialize(pubsubMsg.as_right_adjacent_lanes[i]))

        as_downstream_lanes = list()
        for i in range(pubsubMsg.e_Cnt_downstream_lane_count):
            as_downstream_lanes.append(LaneSegmentConnectivity.deserialize(pubsubMsg.as_downstream_lanes[i]))

        as_upstream_lanes = list()
        for i in range(pubsubMsg.e_Cnt_upstream_lane_count):
            as_upstream_lanes.append(LaneSegmentConnectivity.deserialize(pubsubMsg.as_upstream_lanes[i]))

        as_lane_coupling = list()
        for i in range(pubsubMsg.e_Cnt_lane_coupling_count):
            as_lane_coupling.append(LaneCoupling.deserialize(pubsubMsg.as_lane_coupling[i]))

        # TODO: remove hack of constant MapLaneType after SceneProvider fix
        return cls(pubsubMsg.e_i_lane_segment_id, pubsubMsg.e_i_road_segment_id, MapLaneType(5),
                   pubsubMsg.e_Cnt_static_traffic_flow_control_count, as_static_traffic_flow_control,
                   pubsubMsg.e_Cnt_dynamic_traffic_flow_control_count, as_dynamic_traffic_flow_control,
                   pubsubMsg.e_Cnt_left_adjacent_lane_count, as_left_adjacent_lanes,
                   pubsubMsg.e_Cnt_right_adjacent_lane_count, as_right_adjacent_lanes,
                   pubsubMsg.e_Cnt_downstream_lane_count, as_downstream_lanes,
                   pubsubMsg.e_Cnt_upstream_lane_count, as_upstream_lanes,
                   pubsubMsg.e_v_nominal_speed,
                   pubsubMsg.e_i_downstream_road_intersection_id,
                   pubsubMsg.e_Cnt_lane_coupling_count, as_lane_coupling)


class DataSceneStaticLite(PUBSUB_MSG_IMPL):
    e_b_Valid = bool
    s_RecvTimestamp = Timestamp
    s_ComputeTimestamp = Timestamp
    e_l_perception_horizon_front = float
    e_l_perception_horizon_rear = float
    e_Cnt_num_lane_segments = int
    as_scene_lane_segment = dict[SceneLaneSegmentLite]
    e_Cnt_num_road_intersections = int
    as_scene_road_intersection = List[SceneRoadIntersection]
    e_Cnt_num_road_segments = int
    as_scene_road_segment = List[SceneRoadSegment]

    def __init__(self, e_b_Valid: bool, s_RecvTimestamp:Timestamp, s_ComputeTimestamp: Timestamp, e_l_perception_horizon_front: float,
                 e_l_perception_horizon_rear: float,
                 e_Cnt_num_lane_segments: int, as_scene_lane_segment: List[SceneLaneSegmentLite],
                 e_Cnt_num_road_intersections: int, as_scene_road_intersection: List[SceneRoadIntersection],
                 e_Cnt_num_road_segments: int, as_scene_road_segment: List[SceneRoadSegment]):
        """
        Scene provider's static scene information
        :param e_b_Valid:
        :param s_ComputeTimestamp:
        :param e_l_perception_horizon_front: (Not relevant for M0)
        :param e_l_perception_horizon_rear: (Not relevant for M0)
        :param e_Cnt_num_lane_segments: Total number of lane-segments in the static scene
        :param as_scene_lane_segment: All lane-segments in the static scene
        :param e_Cnt_num_road_intersections: Total number of road-intersections in the static scene
        :param as_scene_road_intersection: All road-intersections in the static scene
        :param e_Cnt_num_road_segments: Total number of road-segments in the static scene
        :param as_scene_road_segment: All road-segments in the static scene
        """
        self.e_b_Valid = e_b_Valid
        self.s_RecvTimestamp = s_RecvTimestamp
        self.s_ComputeTimestamp = s_ComputeTimestamp
        self.e_l_perception_horizon_front = e_l_perception_horizon_front
        self.e_l_perception_horizon_rear = e_l_perception_horizon_rear
        self.e_Cnt_num_lane_segments = e_Cnt_num_lane_segments
        self.as_scene_lane_segment = as_scene_lane_segment
        self.e_Cnt_num_road_intersections = e_Cnt_num_road_intersections
        self.as_scene_road_intersection = as_scene_road_intersection
        self.e_Cnt_num_road_segments = e_Cnt_num_road_segments
        self.as_scene_road_segment = as_scene_road_segment



    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSDataSceneStatic): # should be replaced with TsSYSDataSceneStaticLite

        lane_segments = list()
        for i in range(pubsubMsg.e_Cnt_num_lane_segments_lite):
            lane_segments.append(SceneLaneSegmentLite.deserialize(pubsubMsg.as_scene_lane_segments_lite[i]))

        road_intersections = list()
        for i in range(pubsubMsg.e_Cnt_num_road_intersections):
            road_intersections.append(SceneRoadIntersection.deserialize(pubsubMsg.as_scene_road_intersection[i]))

        road_segments = list()
        for i in range(pubsubMsg.e_Cnt_num_road_segments):
            road_segments.append(SceneRoadSegment.deserialize(pubsubMsg.as_scene_road_segment[i]))

        return cls(pubsubMsg.e_b_Valid, Timestamp.deserialize(pubsubMsg.s_RecvTimestamp),
                   Timestamp.deserialize(pubsubMsg.s_ComputeTimestamp),
                   pubsubMsg.e_l_perception_horizon_front, pubsubMsg.e_l_perception_horizon_rear,
                   pubsubMsg.e_Cnt_num_lane_segments, lane_segments,
                   pubsubMsg.e_Cnt_num_road_intersections, road_intersections,
                   pubsubMsg.e_Cnt_num_road_segments, road_segments)


class SceneStaticLite(PUBSUB_MSG_IMPL):
    s_Header = Header
    s_MapOrigin = MapOrigin
    s_Data = DataSceneStaticLite

    def __init__(self, s_Header: Header, s_MapOrigin: MapOrigin, s_Data: DataSceneStaticLite):
        self.s_Header = s_Header
        self.s_MapOrigin = s_MapOrigin
        self.s_Data = s_Data

    def serialize(self) -> TsSYSSceneStatic:
        pubsub_msg = TsSYSSceneStatic()
        pubsub_msg.s_Header = self.s_Header.serialize()
        pubsub_msg.s_MapOrigin = self.s_MapOrigin.serialize()
        pubsub_msg.s_Data = self.s_Data.serialize()
        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSSceneStatic):
        return cls(Header.deserialize(pubsubMsg.s_Header),
                   MapOrigin.deserialize(pubsubMsg.s_MapOrigin),
                   DataSceneStaticLite.deserialize(pubsubMsg.s_Data))
