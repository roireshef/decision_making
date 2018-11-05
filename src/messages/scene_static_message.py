from enum import Enum
from typing import List
from numpy import np

from common_data.interface.py.idl_generated_files.Rte_Types.sub_structures.TsSYS_SceneRoadIntersection import \
    TsSYSSceneRoadIntersection
from common_data.interface.py.idl_generated_files.Rte_Types.sub_structures.TsSYS_SceneRoadSegment import \
    TsSYSSceneRoadSegment
from common_data.lcm.generatedFiles.gm_lcm import LcmNumpyArray
from decision_making.src.global_constants import PUBSUB_MSG_IMPL
from common_data.interface.py.idl_generated_files.Rte_Types.TsSYS_SceneStatic import TsSYSSceneStatic
from decision_making.src.messages.scene_dynamic_message import Timestamp

# MAX_SCENE_LANE_SEGMENTS = 128
# MAX_SCENE_ROAD_INTERSECTIONS = 64
# MAX_SCENE_ROAD_SEGMENTS = 64


class MapRoadSegmentType(Enum):
   e_MapRoadType_Normal = 0,
   e_MapRoadType_Intersection = 1,
   e_MapRoadType_TurnOnly = 2,
   e_MapRoadType_Unknown = 3


class SceneRoadSegment(PUBSUB_MSG_IMPL):
    e_Cnt_road_segment_id = int
    e_Cnt_road_id = int
    e_Cnt_lane_segment_id_count = int
    a_Cnt_lane_segment_id = List[int]
    e_e_road_segment_type = MapRoadSegmentType
    e_Cnt_upstream_segment_count = int
    a_Cnt_upstream_road_segment_id = List[int]
    e_Cnt_downstream_segment_count = int
    a_Cnt_downstream_road_segment_id = List[int]

    def __init__(self, e_Cnt_road_segment_id, e_Cnt_road_id, e_Cnt_lane_segment_id_count, a_Cnt_lane_segment_id,
                 e_e_road_segment_type, e_Cnt_upstream_segment_count, a_Cnt_upstream_road_segment_id,
                 e_Cnt_downstream_segment_count, a_Cnt_downstream_road_segment_id):
        # type: (int, int, int, List[int], MapRoadSegmentType, int, List[int], int, List[int]) -> None
        self.e_Cnt_road_segment_id = e_Cnt_road_segment_id
        self.e_Cnt_road_id = e_Cnt_road_id
        self.e_Cnt_lane_segment_id_count = e_Cnt_lane_segment_id_count
        self.a_Cnt_lane_segment_id = a_Cnt_lane_segment_id
        self.e_e_road_segment_type = e_e_road_segment_type
        self.e_Cnt_upstream_segment_count = e_Cnt_upstream_segment_count
        self.a_Cnt_upstream_road_segment_id = a_Cnt_upstream_road_segment_id
        self.e_Cnt_downstream_segment_count = e_Cnt_downstream_segment_count
        self.a_Cnt_downstream_road_segment_id = a_Cnt_downstream_road_segment_id

    def serialize(self):
        # type: () -> TsSYSSceneRoadSegment
        pubsub_msg = TsSYSSceneRoadSegment()

        pubsub_msg.e_Cnt_road_segment_id = self.e_Cnt_road_segment_id
        pubsub_msg.e_Cnt_road_id = self.e_Cnt_road_id

        pubsub_msg.e_Cnt_lane_segment_id_count = self.e_Cnt_lane_segment_id_count
        pubsub_msg.a_Cnt_lane_segment_id = LcmNumpyArray()
        pubsub_msg.a_Cnt_lane_segment_id.num_dimensions = len(self.a_Cnt_lane_segment_id.shape)
        pubsub_msg.a_Cnt_lane_segment_id.shape = list(self.a_Cnt_lane_segment_id.shape)
        pubsub_msg.a_Cnt_lane_segment_id.length = self.a_Cnt_lane_segment_id.size
        pubsub_msg.a_Cnt_lane_segment_id.data = self.a_Cnt_lane_segment_id.flat.__array__().tolist()

        pubsub_msg.e_e_road_segment_type = self.e_e_road_segment_type

        pubsub_msg.e_Cnt_upstream_segment_count = self.e_Cnt_upstream_segment_count
        pubsub_msg.a_Cnt_upstream_road_segment_id = LcmNumpyArray()
        pubsub_msg.a_Cnt_upstream_road_segment_id.num_dimensions = len(self.a_Cnt_upstream_road_segment_id.shape)
        pubsub_msg.a_Cnt_upstream_road_segment_id.shape = list(self.a_Cnt_upstream_road_segment_id.shape)
        pubsub_msg.a_Cnt_upstream_road_segment_id.length = self.a_Cnt_upstream_road_segment_id.size
        pubsub_msg.a_Cnt_upstream_road_segment_id.data = self.a_Cnt_upstream_road_segment_id.flat.__array__().tolist()

        pubsub_msg.e_Cnt_downstream_segment_count = self.e_Cnt_downstream_segment_count
        pubsub_msg.a_Cnt_downstream_road_segment_id = LcmNumpyArray()
        pubsub_msg.a_Cnt_downstream_road_segment_id.num_dimensions = len(self.a_Cnt_upstream_road_segment_id.shape)
        pubsub_msg.a_Cnt_downstream_road_segment_id.shape = list(self.a_Cnt_upstream_road_segment_id.shape)
        pubsub_msg.a_Cnt_downstream_road_segment_id.length = self.a_Cnt_upstream_road_segment_id.size
        pubsub_msg.a_Cnt_downstream_road_segment_id.data = self.a_Cnt_upstream_road_segment_id.flat.__array__().tolist()

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg):
        # type: (TsSYSSceneRoadSegment)-> SceneRoadSegment
        return cls(pubsubMsg.e_Cnt_road_segment_id,
                   pubsubMsg.e_Cnt_road_id,
                   pubsubMsg.e_Cnt_lane_segment_id_count,
                   np.ndarray(shape=tuple(pubsubMsg.a_Cnt_lane_segment_id.shape[
                                          :pubsubMsg.a_Cnt_lane_segment_id.num_dimensions])
                              , buffer=np.array(pubsubMsg.a_Cnt_lane_segment_id.data)
                              , dtype=int),
                   pubsubMsg.e_e_road_segment_type,
                   pubsubMsg.e_Cnt_upstream_segment_count,
                   np.ndarray(shape=tuple(pubsubMsg.a_Cnt_upstream_road_segment_id.shape[
                                          :pubsubMsg.a_Cnt_upstream_road_segment_id.num_dimensions])
                              , buffer=np.array(pubsubMsg.a_Cnt_upstream_road_segment_id.data)
                              , dtype=int),
                   pubsubMsg.e_Cnt_downstream_segment_count,
                   np.ndarray(shape=tuple(pubsubMsg.a_Cnt_downstream_road_segment_id.shape[
                                          :pubsubMsg.a_Cnt_downstream_road_segment_id.num_dimensions])
                              , buffer=np.array(pubsubMsg.a_Cnt_downstream_road_segment_id.data)
                              , dtype=int))


class SceneRoadIntersection(PUBSUB_MSG_IMPL):
    e_i_road_intersection_id = int
    e_Cnt_lane_coupling_count = int
    a_i_lane_coupling_segment_ids = List[int]
    e_Cnt_intersection_road_segment_count = int
    a_i_intersection_road_segment_ids = List[int]

    def __init__(self, e_i_road_intersection_id, e_Cnt_lane_coupling_count, a_i_lane_coupling_segment_ids,
                 e_Cnt_intersection_road_segment_count, a_i_intersection_road_segment_ids):
        # type (int, int, List[int], int, List[int]) -> None
        self.e_i_road_intersection_id = e_i_road_intersection_id
        self.e_Cnt_lane_coupling_count = e_Cnt_lane_coupling_count
        self.a_i_lane_coupling_segment_ids = a_i_lane_coupling_segment_ids
        self.e_Cnt_intersection_road_segment_count = e_Cnt_intersection_road_segment_count
        self.a_i_intersection_road_segment_ids = a_i_intersection_road_segment_ids

    def serialize(self):
        # type: () -> TsSYSSceneRoadIntersection
        pubsub_msg = TsSYSSceneRoadIntersection()

        pubsub_msg.e_i_road_intersection_id = self.e_i_road_intersection_id
        pubsub_msg.e_Cnt_lane_coupling_count = self.e_Cnt_lane_coupling_count
        pubsub_msg.a_i_lane_coupling_segment_ids = LcmNumpyArray()
        pubsub_msg.a_i_lane_coupling_segment_ids.num_dimensions = len(self.a_i_lane_coupling_segment_ids.shape)
        pubsub_msg.a_i_lane_coupling_segment_ids.shape = list(self.a_i_lane_coupling_segment_ids.shape)
        pubsub_msg.a_i_lane_coupling_segment_ids.length = self.a_i_lane_coupling_segment_ids.size
        pubsub_msg.a_i_lane_coupling_segment_ids.data = self.a_i_lane_coupling_segment_ids.flat.__array__().tolist()

        pubsub_msg.e_Cnt_intersection_road_segment_count = self.e_Cnt_intersection_road_segment_count
        pubsub_msg.a_i_intersection_road_segment_ids = LcmNumpyArray()
        pubsub_msg.a_i_intersection_road_segment_ids.num_dimensions = len(self.a_i_intersection_road_segment_ids.shape)
        pubsub_msg.a_i_intersection_road_segment_ids.shape = list(self.a_i_intersection_road_segment_ids.shape)
        pubsub_msg.a_i_intersection_road_segment_ids.length = self.a_i_intersection_road_segment_ids.size
        pubsub_msg.a_i_intersection_road_segment_ids.data = self.a_i_intersection_road_segment_ids.flat.__array__().tolist()

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg):
        # type: (TsSYSSceneRoadIntersection) -> SceneRoadIntersection
        return cls(pubsubMsg.e_i_road_intersection_id,
                   pubsubMsg.e_Cnt_lane_coupling_count,
                   np.ndarray(shape=tuple(pubsubMsg.a_i_lane_coupling_segment_ids.shape[
                                          :pubsubMsg.a_i_lane_coupling_segment_ids.num_dimensions])
                              , buffer=np.array(pubsubMsg.a_i_lane_coupling_segment_ids.data)
                              , dtype=int),
                   pubsubMsg.e_Cnt_intersection_road_segment_count,
                   np.ndarray(shape=tuple(pubsubMsg.a_i_intersection_road_segment_ids.shape[
                                          :pubsubMsg.a_i_intersection_road_segment_ids.num_dimensions])
                              , buffer=np.array(pubsubMsg.a_i_intersection_road_segment_ids.data)
                              , dtype=int))


class SceneStatic(PUBSUB_MSG_IMPL):
    e_b_Valid = bool
    s_ComputeTimestamp = Timestamp
    e_l_perception_horizon_front = float
    e_l_perception_horizon_rear = float
    e_Cnt_num_lane_segments = int
    as_scene_lane_segment = List[SceneLaneSegment]
    e_Cnt_num_road_intersections = int
    as_scene_road_intersection = List[SceneRoadIntersection]
    e_Cnt_num_road_segments = int
    as_scene_road_segment = List[SceneRoadSegment]

    def __init__(self, e_b_Valid, s_ComputeTimestamp, e_l_perception_horizon_front, e_l_perception_horizon_rear,
                 e_Cnt_num_lane_segments, as_scene_lane_segment, e_Cnt_num_road_intersections,
                 as_scene_road_intersection, e_Cnt_num_road_segments, as_scene_road_segment):
        # type: (bool, Timestamp, float, float, int, List[SceneLaneSegment], int, List[SceneRoadIntersection], int, List[SceneRoadSegment]) -> None
        self.e_b_Valid = e_b_Valid
        self.s_ComputeTimestamp = s_ComputeTimestamp
        self.e_l_perception_horizon_front = e_l_perception_horizon_front
        self.e_l_perception_horizon_rear = e_l_perception_horizon_rear
        self.e_Cnt_num_lane_segments = e_Cnt_num_lane_segments
        self.as_scene_lane_segment = as_scene_lane_segment
        self.e_Cnt_num_road_intersections = e_Cnt_num_road_intersections
        self.as_scene_road_intersection = as_scene_road_intersection
        self.e_Cnt_num_road_segments = e_Cnt_num_road_segments
        self.as_scene_road_segment = as_scene_road_segment

    def serialize(self):
        # type: () -> TsSYSSceneStatic
        pubsub_msg = TsSYSSceneStatic()

        pubsub_msg.e_b_Valid = self.e_b_Valid
        pubsub_msg.s_ComputeTimestamp = self.s_ComputeTimestamp.serialize()
        pubsub_msg.e_l_perception_horizon_front = self.e_l_perception_horizon_front
        pubsub_msg.e_l_perception_horizon_rear = self.e_l_perception_horizon_rear

        pubsub_msg.e_Cnt_num_lane_segments = self.e_Cnt_num_lane_segments
        pubsub_msg.as_scene_lane_segment = list()
        for i in range(pubsub_msg.e_Cnt_num_lane_segments):
            pubsub_msg.as_scene_lane_segment.append(self.as_scene_lane_segment[i].serialize())

        pubsub_msg.e_Cnt_num_road_intersections = self.e_Cnt_num_road_intersections
        pubsub_msg.as_scene_road_intersection = list()
        for i in range(pubsub_msg.e_Cnt_num_road_intersections):
            pubsub_msg.as_scene_road_intersection.append(self.as_scene_road_intersection[i].serialize())

        pubsub_msg.e_Cnt_num_road_segments = self.e_Cnt_num_road_segments
        pubsub_msg.as_scene_road_segment = list()
        for i in range(pubsub_msg.e_Cnt_num_road_segments):
            pubsub_msg.as_scene_road_segment.append(self.as_scene_road_segment[i].serialize())

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg):
        # type: (TsSYSSceneStatic) -> SceneStatic

        lane_segments = list()
        for i in range(pubsubMsg.e_Cnt_num_lane_segments):
            lane_segments.append(SceneLaneSegment.deserialize(pubsubMsg.as_scene_lane_segment[i]))

        road_intersections = list()
        for i in range(pubsubMsg.e_Cnt_num_road_intersections):
            road_intersections.append(SceneRoadIntersection.deserialize(pubsubMsg.as_scene_road_intersection[i]))

        road_segments = list()
        for i in range(pubsubMsg.e_Cnt_num_road_segments):
            road_segments.append(SceneRoadSegment.deserialize(pubsubMsg.as_scene_road_segment[i]))

        return cls(pubsubMsg.e_b_Valid, Timestamp.deserialize(pubsubMsg.s_ComputeTimestamp),
                   pubsubMsg.e_l_perception_horizon_front, pubsubMsg.e_l_perception_horizon_rear,
                   pubsubMsg.e_Cnt_num_lane_segments, lane_segments,
                   pubsubMsg.e_Cnt_num_road_intersections, road_intersections,
                   pubsubMsg.e_Cnt_num_road_segments, road_segments)
