from enum import IntEnum
from typing import List
import numpy as np
from interface.Rte_Types.python.sub_structures.TsSYS_AdjacentLane import TsSYSAdjacentLane
from interface.Rte_Types.python.sub_structures.TsSYS_BoundaryPoint import TsSYSBoundaryPoint
from interface.Rte_Types.python.sub_structures.TsSYS_StaticTrafficControlDevice import TsSYSStaticTrafficControlDevice
from interface.Rte_Types.python.sub_structures.TsSYS_DynamicTrafficControlDevice import TsSYSDynamicTrafficControlDevice
from interface.Rte_Types.python.sub_structures.TsSYS_LaneSegmentConnectivity import TsSYSLaneSegmentConnectivity
from interface.Rte_Types.python.sub_structures.TsSYS_SceneLaneSegmentBase import TsSYSSceneLaneSegmentBase
from interface.Rte_Types.python.sub_structures.TsSYS_SceneLaneSegmentGeometry import TsSYSSceneLaneSegmentGeometry
from interface.Rte_Types.python.sub_structures.TsSYS_NavigationPlan import TsSYSNavigationPlan
from interface.Rte_Types.python.sub_structures.TsSYS_SceneStaticGeometry import TsSYSSceneStaticGeometry
from interface.Rte_Types.python.sub_structures.TsSYS_SceneStaticBase import TsSYSSceneStaticBase
from interface.Rte_Types.python.sub_structures.TsSYS_SceneStatic import TsSYSSceneStatic
from interface.Rte_Types.python.sub_structures.TsSYS_DataSceneStatic import TsSYSDataSceneStatic
from interface.Rte_Types.python.sub_structures.TsSYS_SceneRoadSegment import TsSYSSceneRoadSegment
from interface.Rte_Types.python.sub_structures.TsSYS_LaneOverlap import TsSYSLaneOverlap
from interface.Rte_Types.python.sub_structures.TsSYS_TrafficControlBar import TsSYSTrafficControlBar
from decision_making.src.messages.serialization import PUBSUB_MSG_IMPL
from decision_making.src.messages.scene_static_enums import RoutePlanLaneSegmentAttr, \
    LaneMappingStatusType, GMAuthorityType, LaneConstructionType, MapLaneDirection
from decision_making.src.messages.scene_common_messages import Timestamp, MapOrigin, Header
from decision_making.src.messages.scene_static_enums import MapLaneType, MapRoadSegmentType, MovingDirection, \
    ManeuverType, MapLaneMarkerType, LaneOverlapType, StaticTrafficControlDeviceType, DynamicTrafficControlDeviceType

MAX_LANE_ATTRIBUTES = 8
MAX_NOMINAL_PATH_POINT_FIELDS = 10


class SceneRoadSegment(PUBSUB_MSG_IMPL):
    e_i_road_segment_id = int
    e_Cnt_lane_segment_id_count = int
    a_i_lane_segment_ids = np.ndarray
    e_e_road_segment_type = MapRoadSegmentType
    e_Cnt_upstream_segment_count = int
    a_i_upstream_road_segment_ids = np.ndarray
    e_Cnt_downstream_segment_count = int
    a_i_downstream_road_segment_ids = np.ndarray

    def __init__(self, e_i_road_segment_id: int, e_Cnt_lane_segment_id_count: int,
                 a_i_lane_segment_ids: np.ndarray, e_e_road_segment_type: MapRoadSegmentType,
                 e_Cnt_upstream_segment_count: int, a_i_upstream_road_segment_ids: np.ndarray,
                 e_Cnt_downstream_segment_count: int, a_i_downstream_road_segment_ids: np.ndarray) -> None:
        """
        Road-segment information
        :param e_i_road_segment_id: ID of this Road-segment
        :param e_Cnt_lane_segment_id_count: Total number of all lane-segments contained within this road-segment
        :param a_i_lane_segment_ids: Lane-segments contained within this road-segment
        :param e_e_road_segment_type:
        :param e_Cnt_upstream_segment_count: Total number of upstream road-segments from this road-segment
        :param a_i_upstream_road_segment_ids: Upstream road-segments from this road-segment
        :param e_Cnt_downstream_segment_count: Total number of downstream road-segments from this road-segment
        :param a_i_downstream_road_segment_ids: Downstream road-segments from this road-segment
        """
        self.e_i_road_segment_id = e_i_road_segment_id
        self.e_Cnt_lane_segment_id_count = e_Cnt_lane_segment_id_count
        self.a_i_lane_segment_ids = a_i_lane_segment_ids
        self.e_e_road_segment_type = e_e_road_segment_type
        self.e_Cnt_upstream_segment_count = e_Cnt_upstream_segment_count
        self.a_i_upstream_road_segment_ids = a_i_upstream_road_segment_ids
        self.e_Cnt_downstream_segment_count = e_Cnt_downstream_segment_count
        self.a_i_downstream_road_segment_ids = a_i_downstream_road_segment_ids

    def serialize(self) -> TsSYSSceneRoadSegment:
        pubsub_msg = TsSYSSceneRoadSegment()

        pubsub_msg.e_i_road_segment_id = self.e_i_road_segment_id

        pubsub_msg.e_Cnt_lane_segment_id_count = self.e_Cnt_lane_segment_id_count
        pubsub_msg.a_i_lane_segment_ids = self.a_i_lane_segment_ids

        pubsub_msg.e_e_road_segment_type = self.e_e_road_segment_type.value

        pubsub_msg.e_Cnt_upstream_segment_count = self.e_Cnt_upstream_segment_count
        pubsub_msg.a_i_upstream_road_segment_ids = self.a_i_upstream_road_segment_ids

        pubsub_msg.e_Cnt_downstream_segment_count = self.e_Cnt_downstream_segment_count
        pubsub_msg.a_i_downstream_road_segment_ids = self.a_i_downstream_road_segment_ids

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSSceneRoadSegment):
        return cls(pubsubMsg.e_i_road_segment_id,
                   pubsubMsg.e_Cnt_lane_segment_id_count,
                   pubsubMsg.a_i_lane_segment_ids[:pubsubMsg.e_Cnt_lane_segment_id_count],
                   MapRoadSegmentType(pubsubMsg.e_e_road_segment_type),
                   pubsubMsg.e_Cnt_upstream_segment_count,
                   pubsubMsg.a_i_upstream_road_segment_ids[:pubsubMsg.e_Cnt_upstream_segment_count],
                   pubsubMsg.e_Cnt_downstream_segment_count,
                   pubsubMsg.a_i_downstream_road_segment_ids[:pubsubMsg.e_Cnt_downstream_segment_count])


class AdjacentLane(PUBSUB_MSG_IMPL):
    e_i_lane_segment_id = int
    e_e_moving_direction = MovingDirection
    e_e_lane_type = MapLaneType

    def __init__(self, e_i_lane_segment_id: int, e_e_moving_direction: MovingDirection, e_e_lane_type: MapLaneType):
        self.e_i_lane_segment_id = e_i_lane_segment_id
        self.e_e_moving_direction = e_e_moving_direction
        self.e_e_lane_type = e_e_lane_type

    def serialize(self) -> TsSYSAdjacentLane:
        pubsub_msg = TsSYSAdjacentLane()

        pubsub_msg.e_i_lane_segment_id = self.e_i_lane_segment_id
        pubsub_msg.e_e_moving_direction = self.e_e_moving_direction.value
        pubsub_msg.e_e_lane_type = self.e_e_lane_type.value

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSAdjacentLane):
        return cls(pubsubMsg.e_i_lane_segment_id, MovingDirection(pubsubMsg.e_e_moving_direction),
                   MapLaneType(pubsubMsg.e_e_lane_type))


class LaneSegmentConnectivity(PUBSUB_MSG_IMPL):
    e_i_lane_segment_id = int
    e_e_maneuver_type = ManeuverType

    def __init__(self, e_i_lane_segment_id: int, e_e_maneuver_type: ManeuverType):
        self.e_i_lane_segment_id = e_i_lane_segment_id
        self.e_e_maneuver_type = e_e_maneuver_type

    def serialize(self) -> TsSYSLaneSegmentConnectivity:
        pubsub_msg = TsSYSLaneSegmentConnectivity()

        pubsub_msg.e_i_lane_segment_id = self.e_i_lane_segment_id
        pubsub_msg.e_e_maneuver_type = self.e_e_maneuver_type.value

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSLaneSegmentConnectivity):
        return cls(pubsubMsg.e_i_lane_segment_id, ManeuverType(pubsubMsg.e_e_maneuver_type))


class LaneOverlap(PUBSUB_MSG_IMPL):
    e_i_other_lane_segment_id = int
    a_l_source_lane_overlap_stations = np.ndarray
    a_l_other_lane_overlap_stations = np.ndarray
    e_e_lane_overlap_type = LaneOverlapType

    def __init__(self, e_i_other_lane_segment_id: int, a_l_source_lane_overlap_stations: np.ndarray,
                 a_l_other_lane_overlap_stations: np.ndarray, e_e_lane_overlap_type: LaneOverlapType):
        self.e_i_other_lane_segment_id = e_i_other_lane_segment_id
        self.a_l_source_lane_overlap_stations = a_l_source_lane_overlap_stations
        self.a_l_other_lane_overlap_stations = a_l_other_lane_overlap_stations
        self.e_e_lane_overlap_type = e_e_lane_overlap_type

    def serialize(self) -> TsSYSLaneOverlap:
        pubsub_msg = TsSYSLaneOverlap()

        pubsub_msg.e_i_other_lane_segment_id = self.e_i_other_lane_segment_id
        pubsub_msg.a_l_source_lane_overlap_stations = self.a_l_source_lane_overlap_stations
        pubsub_msg.a_l_other_lane_overlap_stations = self.a_l_other_lane_overlap_stations
        pubsub_msg.e_e_lane_overlap_type = self.e_e_lane_overlap_type

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSLaneOverlap):
        return cls(pubsubMsg.e_i_other_lane_segment_id, pubsubMsg.a_l_source_lane_overlap_stations,
                   pubsubMsg.a_l_other_lane_overlap_stations,
                   LaneOverlapType(pubsubMsg.e_e_lane_overlap_type))


class BoundaryPoint(PUBSUB_MSG_IMPL):
    e_e_lane_marker_type = MapLaneMarkerType
    e_l_s_start = float
    e_l_s_end = float

    def __init__(self, e_e_lane_marker_type: MapLaneMarkerType, e_l_s_start: float, e_l_s_end: float):
        self.e_e_lane_marker_type = e_e_lane_marker_type
        self.e_l_s_start = e_l_s_start
        self.e_l_s_end = e_l_s_end

    def serialize(self) -> TsSYSBoundaryPoint:
        pubsub_msg = TsSYSBoundaryPoint()

        pubsub_msg.e_e_lane_marker_type = self.e_e_lane_marker_type.value
        pubsub_msg.e_l_s_start = self.e_l_s_start
        pubsub_msg.e_l_s_end = self.e_l_s_end

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSBoundaryPoint):
        return cls(pubsubMsg.e_e_lane_marker_type, pubsubMsg.e_l_s_start, pubsubMsg.e_l_s_end)


class TrafficControlBar(PUBSUB_MSG_IMPL):
    e_i_traffic_control_bar_id = int
    e_l_station = float
    e_i_static_traffic_control_device_id = List[int]
    e_i_dynamic_traffic_control_device_id = List[int]

    def __init__(self, e_i_traffic_control_bar_id: int, e_l_station: float,
                 e_i_static_traffic_control_device_id: List[int], e_i_dynamic_traffic_control_device_id: List[int]):
        """
        Traffic-control-bar i.e stop bar. Either physical or virtual
        :param e_i_traffic_control_bar_id:
        :param e_l_station:
        :param e_i_static_traffic_control_device_id:
        :param e_i_dynamic_traffic_control_device_id:
        """
        self.e_i_traffic_control_bar_id = e_i_traffic_control_bar_id
        self.e_l_station = e_l_station
        self.e_i_static_traffic_control_device_id = e_i_static_traffic_control_device_id
        self.e_i_dynamic_traffic_control_device_id = e_i_dynamic_traffic_control_device_id

    def serialize(self) -> TsSYSTrafficControlBar:
        pubsub_msg = TsSYSTrafficControlBar()

        pubsub_msg.e_i_traffic_control_bar_id = self.e_i_traffic_control_bar_id
        pubsub_msg.e_l_station = self.e_l_station
        pubsub_msg.e_Cnt_static_traffic_control_device_count = len(self.e_i_static_traffic_control_device_id)
        for i in range(pubsub_msg.e_Cnt_static_traffic_control_device_count):
            pubsub_msg.e_i_static_traffic_control_device_id[i] = self.e_i_static_traffic_control_device_id[i]
        pubsub_msg.e_Cnt_dynamic_traffic_control_device_count = len(self.e_i_dynamic_traffic_control_device_id)
        for i in range(pubsub_msg.e_Cnt_dynamic_traffic_control_device_count):
            pubsub_msg.e_i_dynamic_traffic_control_device_id[i] = self.e_i_dynamic_traffic_control_device_id[i]

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSTrafficControlBar):
        e_i_static_traffic_control_device_id = list()
        for i in range(pubsubMsg.e_Cnt_static_traffic_control_device_count):
            e_i_static_traffic_control_device_id.append(pubsubMsg.e_i_static_traffic_control_device_id[i])
        e_i_dynamic_traffic_control_device_id = list()
        for i in range(pubsubMsg.e_Cnt_dynamic_traffic_control_device_count):
            e_i_dynamic_traffic_control_device_id.append(pubsubMsg.e_i_dynamic_traffic_control_device_id[i])

        return cls(pubsubMsg.e_i_traffic_control_bar_id, pubsubMsg.e_l_station,
                   e_i_static_traffic_control_device_id, e_i_dynamic_traffic_control_device_id)


class StaticTrafficControlDevice(PUBSUB_MSG_IMPL):
    object_id = int
    e_e_traffic_control_device_type = StaticTrafficControlDeviceType
    e_Pct_confidence = float
    e_i_controlled_lane_segment_id = List[int]
    e_l_east_x = float
    e_l_north_y = float

    def __init__(self, object_id: int, e_e_traffic_control_device_type: StaticTrafficControlDeviceType, e_Pct_confidence: float,
                 e_i_controlled_lane_segment_id: List[int], e_l_east_x: float, e_l_north_y: float):
        """
        Static traffic-flow-control device, eg. Stop Signs (not relevant for M0)
        :param object_id:
        :param e_e_traffic_control_device_type:
        :param e_Pct_confidence:
        :param e_i_controlled_lane_segment_id
        :param e_l_east_x:
        :param e_l_north_y:
        """
        self.object_id = object_id
        self.e_e_traffic_control_device_type = e_e_traffic_control_device_type
        self.e_Pct_confidence = e_Pct_confidence
        self.e_i_controlled_lane_segment_id = e_i_controlled_lane_segment_id
        self.e_l_east_x = e_l_east_x
        self.e_l_north_y = e_l_north_y

    def serialize(self) -> TsSYSStaticTrafficControlDevice:
        pubsub_msg = TsSYSStaticTrafficControlDevice()

        pubsub_msg.e_i_static_traffic_control_device_id = self.object_id
        pubsub_msg.e_e_traffic_control_device_type = self.e_e_traffic_control_device_type.value
        pubsub_msg.e_Pct_confidence = self.e_Pct_confidence
        pubsub_msg.e_Cnt_controlled_lane_segments_count = len(self.e_i_controlled_lane_segment_id)
        for i in range(len(self.e_i_controlled_lane_segment_id)):
            pubsub_msg.e_i_controlled_lane_segment_id[i] = self.e_i_controlled_lane_segment_id[i]
        pubsub_msg.e_l_east_x = self.e_l_east_x
        pubsub_msg.e_l_north_y = self.e_l_north_y

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSStaticTrafficControlDevice):
        e_i_controlled_lane_segment_id = list()
        for i in range(pubsubMsg.e_Cnt_controlled_lane_segments_count):
            e_i_controlled_lane_segment_id.append(pubsubMsg.e_i_controlled_lane_segment_id[i])

        return cls(pubsubMsg.e_i_static_traffic_control_device_id,
                   StaticTrafficControlDeviceType(pubsubMsg.e_e_traffic_control_device_type),
                   pubsubMsg.e_Pct_confidence, e_i_controlled_lane_segment_id, pubsubMsg.e_l_east_x,
                   pubsubMsg.e_l_north_y)


class DynamicTrafficControlDevice(PUBSUB_MSG_IMPL):
    object_id = int
    e_e_traffic_control_device_type = DynamicTrafficControlDeviceType
    e_i_controlled_lane_segment_id = List[int]
    e_l_east_x = float
    e_l_north_y = float

    def __init__(self, object_id: int, e_e_traffic_control_device_type: DynamicTrafficControlDeviceType,
                 e_i_controlled_lane_segment_id: List[int], e_l_east_x: float, e_l_north_y: float):
        """
        Dynamic traffic-flow-control device, e.g. Traffic lights (not relevant for M0)
        :param object_id:
        :param e_e_traffic_control_device_type:
        :param e_l_east_x:
        :param e_l_north_y:
        """
        self.object_id = object_id
        self.e_e_traffic_control_device_type = e_e_traffic_control_device_type
        self.e_i_controlled_lane_segment_id = e_i_controlled_lane_segment_id
        self.e_l_east_x = e_l_east_x
        self.e_l_north_y = e_l_north_y

    def serialize(self) -> TsSYSDynamicTrafficControlDevice:
        pubsub_msg = TsSYSDynamicTrafficControlDevice()

        pubsub_msg.e_i_dynamic_traffic_control_device_id = self.object_id
        pubsub_msg.e_e_traffic_control_device_type = self.e_e_traffic_control_device_type.value
        pubsub_msg.e_Cnt_controlled_lane_segments_count = len(self.e_i_controlled_lane_segment_id)
        for i in range(len(self.e_i_controlled_lane_segment_id)):
            pubsub_msg.e_i_controlled_lane_segment_id[i] = self.e_i_controlled_lane_segment_id[i]
        pubsub_msg.e_l_east_x = self.e_l_east_x
        pubsub_msg.e_l_north_y = self.e_l_north_y

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSDynamicTrafficControlDevice):
        e_i_controlled_lane_segment_id = list()
        for i in range(pubsubMsg.e_Cnt_controlled_lane_segments_count):
            e_i_controlled_lane_segment_id.append(pubsubMsg.e_i_controlled_lane_segment_id[i])

        return cls(pubsubMsg.e_i_dynamic_traffic_control_device_id,
                   DynamicTrafficControlDeviceType(pubsubMsg.e_e_traffic_control_device_type),
                   e_i_controlled_lane_segment_id,
                   pubsubMsg.e_l_east_x, pubsubMsg.e_l_north_y)


class SceneLaneSegmentGeometry(PUBSUB_MSG_IMPL):
    e_i_lane_segment_id = int
    e_i_road_segment_id = int
    e_Cnt_nominal_path_point_count = int
    a_nominal_path_points = np.ndarray
    e_Cnt_left_boundary_points_count = int
    as_left_boundary_points = List[BoundaryPoint]
    e_Cnt_right_boundary_points_count = int
    as_right_boundary_points = List[BoundaryPoint]

    def __init__(self, e_i_lane_segment_id: int, e_i_road_segment_id: int, e_Cnt_nominal_path_point_count: int,
                 a_nominal_path_points: np.ndarray,
                 e_Cnt_left_boundary_points_count: int, as_left_boundary_points: List[BoundaryPoint],
                 e_Cnt_right_boundary_points_count: int, as_right_boundary_points: List[BoundaryPoint]):
        """
        Lane-segment information
        :param e_i_lane_segment_id: ID of this lane-segment
        :param e_i_road_segment_id: ID of the road-segment that this lane-segment belongs to
        :param e_Cnt_nominal_path_point_count: Total number of points that specify the nominal-path (i.e. center of lane) for this lane-segment
        :param a_nominal_path_points: Points that specify the nominal-path (i.e. center of lane) for this lane-segment.
               Its shape has to be [e_Cnt_nominal_path_point_count X MAX_NOMINAL_PATH_POINT_FIELDS].
        :param e_Cnt_left_boundary_points_count: Total number of points that specify the left-boundary for this lane-segment
        :param as_left_boundary_points: Points that specify the left-boundary for this lane-segment
        :param e_Cnt_right_boundary_points_count: Total number of points that specify the right-boundary for this lane-segment
        :param as_right_boundary_points: Points that specify the right-boundary for this lane-segment
        """
        self.e_i_lane_segment_id = e_i_lane_segment_id
        self.e_i_road_segment_id = e_i_road_segment_id
        self.e_Cnt_nominal_path_point_count = e_Cnt_nominal_path_point_count
        self.a_nominal_path_points = a_nominal_path_points
        self.e_Cnt_left_boundary_points_count = e_Cnt_left_boundary_points_count
        self.as_left_boundary_points = as_left_boundary_points
        self.e_Cnt_right_boundary_points_count = e_Cnt_right_boundary_points_count
        self.as_right_boundary_points = as_right_boundary_points

    def serialize(self) -> TsSYSSceneLaneSegmentGeometry:
        pubsub_msg = TsSYSSceneLaneSegmentGeometry()

        pubsub_msg.e_i_lane_segment_id = self.e_i_lane_segment_id
        pubsub_msg.e_i_road_segment_id = self.e_i_road_segment_id

        pubsub_msg.e_Cnt_nominal_path_point_count = self.e_Cnt_nominal_path_point_count
        pubsub_msg.a_nominal_path_points = self.a_nominal_path_points

        pubsub_msg.e_Cnt_left_boundary_points_count = self.e_Cnt_left_boundary_points_count
        for i in range(pubsub_msg.e_Cnt_left_boundary_points_count):
            pubsub_msg.as_left_boundary_points[i] = self.as_left_boundary_points[i].serialize()

        pubsub_msg.e_Cnt_right_boundary_points_count = self.e_Cnt_right_boundary_points_count
        for i in range(pubsub_msg.e_Cnt_right_boundary_points_count):
            pubsub_msg.as_right_boundary_points[i] = self.as_right_boundary_points[i].serialize()

        return pubsub_msg

    @classmethod
    def deserialize(cls,
                    pubsubMsg: TsSYSSceneLaneSegmentGeometry,
                    a_nominal_path_points: np.ndarray):

        as_left_boundary_points = list()
        for i in range(pubsubMsg.e_Cnt_left_boundary_points_count):
            as_left_boundary_points.append(BoundaryPoint.deserialize(pubsubMsg.as_left_boundary_points[i]))

        as_right_boundary_points = list()
        for i in range(pubsubMsg.e_Cnt_right_boundary_points_count):
            as_right_boundary_points.append(BoundaryPoint.deserialize(pubsubMsg.as_right_boundary_points[i]))

        # TODO: remove hack of constant MapLaneType after SceneProvider fix
        return cls(pubsubMsg.e_i_lane_segment_id, pubsubMsg.e_i_road_segment_id,
                   pubsubMsg.e_Cnt_nominal_path_point_count, a_nominal_path_points,
                   pubsubMsg.e_Cnt_left_boundary_points_count, as_left_boundary_points,
                   pubsubMsg.e_Cnt_right_boundary_points_count, as_right_boundary_points)


class SceneStaticGeometry(PUBSUB_MSG_IMPL):
    e_Cnt_num_lane_segments = int
    as_scene_lane_segments = List[SceneLaneSegmentGeometry]

    def __init__(self, e_Cnt_num_lane_segments: int,
                 as_scene_lane_segments: List[SceneLaneSegmentGeometry],
                 a_nominal_path_points: np.ndarray):
        """
        Scene provider's static scene information
        :param a_nominal_path_points:
        :param e_Cnt_num_lane_segments: Total number of lane-segments(geometry) in the static scene
        :param as_scene_lane_segments: All lane-segments(geometry) in the static scene
        """
        self.e_Cnt_num_lane_segments = e_Cnt_num_lane_segments
        self.as_scene_lane_segments = as_scene_lane_segments
        self.a_nominal_path_points = a_nominal_path_points

    def serialize(self) -> TsSYSSceneStaticGeometry:
        pubsub_msg = TsSYSSceneStaticGeometry()

        pubsub_msg.e_Cnt_num_lane_segments = self.e_Cnt_num_lane_segments
        pubsub_msg.a_nominal_path_points = self.a_nominal_path_points
        for i in range(pubsub_msg.e_Cnt_num_lane_segments):
            pubsub_msg.as_scene_lane_segments[i] = self.as_scene_lane_segments[i].serialize()

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSSceneStaticGeometry):

        lane_segments_geometry = list()
        for i in range(pubsubMsg.e_Cnt_num_lane_segments):
            # Get the current lane relevant nominal path points
            e_cnt_nominal_path_point_count = pubsubMsg.as_scene_lane_segments[i].e_Cnt_nominal_path_point_count
            # todo: replace "e_i_nominal_path_point_start_index"] with .e_i_nominal_path_point_start_index once conmmon_data integrated

            e_i_nominal_path_point_start_index = pubsubMsg.as_scene_lane_segments[i]._dic["e_i_nominal_path_point_start_index"]
            e_i_nominal_path_point_end_index = e_i_nominal_path_point_start_index + e_cnt_nominal_path_point_count
            curr_lane_a_nominal_path_points = pubsubMsg.a_nominal_path_points[
                                              e_i_nominal_path_point_start_index:e_i_nominal_path_point_end_index, :
                                              ]
            lane_segments_geometry.append(SceneLaneSegmentGeometry.deserialize(pubsubMsg.as_scene_lane_segments[i],
                                                                               curr_lane_a_nominal_path_points))

        return cls(pubsubMsg.e_Cnt_num_lane_segments,
                   lane_segments_geometry,
                   pubsubMsg.a_nominal_path_points)


class NavigationPlan(PUBSUB_MSG_IMPL):
    e_Cnt_num_road_segments = int
    a_i_road_segment_ids = np.ndarray

    def __init__(self, e_Cnt_num_road_segments: int, a_i_road_segment_ids: np.ndarray):
        """
        TODO
        :param e_Cnt_num_road_segments:
        :param a_i_road_segment_ids:
        """
        self.e_Cnt_num_road_segments = e_Cnt_num_road_segments
        self.a_i_road_segment_ids = a_i_road_segment_ids

    def serialize(self) -> TsSYSNavigationPlan:
        pubsub_msg = TsSYSNavigationPlan()

        pubsub_msg.e_Cnt_num_road_segments = self.e_Cnt_num_road_segments
        pubsub_msg.a_i_road_segment_ids = self.a_i_road_segment_ids

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSNavigationPlan):
        return cls(pubsubMsg.e_Cnt_num_road_segments,
                   pubsubMsg.a_i_road_segment_ids[:pubsubMsg.e_Cnt_num_road_segments])


class SceneLaneSegmentBase(PUBSUB_MSG_IMPL):
    e_i_lane_segment_id = int
    e_i_road_segment_id = int
    e_e_lane_type = MapLaneType
    e_Cnt_traffic_control_bar_count = int
    as_traffic_control_bar = List[TrafficControlBar]
    e_Cnt_left_adjacent_lane_count = int
    as_left_adjacent_lanes = List[AdjacentLane]
    e_Cnt_right_adjacent_lane_count = int
    as_right_adjacent_lanes = List[AdjacentLane]
    e_Cnt_downstream_lane_count = int
    as_downstream_lanes = List[LaneSegmentConnectivity]
    e_Cnt_upstream_lane_count = int
    as_upstream_lanes = List[LaneSegmentConnectivity]
    e_v_nominal_speed = float
    e_l_length = float
    e_Cnt_num_active_lane_attributes = int
    a_i_active_lane_attribute_indices = np.ndarray
    a_cmp_lane_attributes = List[IntEnum]
    a_cmp_lane_attribute_confidences = np.ndarray
    e_Cnt_lane_overlap_count = int
    as_lane_overlaps = List[LaneOverlap]

    # These are class variables that are shared between all instances of SceneLaneSegmentBase
    lane_attribute_types = {RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_MappingStatus: LaneMappingStatusType,
                            RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_GMFA: GMAuthorityType,
                            RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_Construction: LaneConstructionType,
                            RoutePlanLaneSegmentAttr.CeSYS_e_RoutePlanLaneSegmentAttr_Direction: MapLaneDirection}
    num_lane_attributes = len(lane_attribute_types)

    def __init__(self, e_i_lane_segment_id: int, e_i_road_segment_id: int, e_e_lane_type: MapLaneType,
                 e_Cnt_traffic_control_bar_count: int,
                 as_traffic_control_bar: List[TrafficControlBar],
                 e_Cnt_left_adjacent_lane_count: int, as_left_adjacent_lanes: List[AdjacentLane],
                 e_Cnt_right_adjacent_lane_count: int, as_right_adjacent_lanes: List[AdjacentLane],
                 e_Cnt_downstream_lane_count: int, as_downstream_lanes: List[LaneSegmentConnectivity],
                 e_Cnt_upstream_lane_count: int, as_upstream_lanes: List[LaneSegmentConnectivity],
                 e_v_nominal_speed: float, e_l_length: float,
                 e_Cnt_num_active_lane_attributes: int, a_i_active_lane_attribute_indices: np.ndarray,
                 a_cmp_lane_attributes: List[IntEnum], a_cmp_lane_attribute_confidences: np.ndarray,
                 e_Cnt_lane_overlap_count: int, as_lane_overlaps: List[LaneOverlap]):
        """
        Lane-segment information
        :param e_i_lane_segment_id: ID of this lane-segment
        :param e_i_road_segment_id: ID of the road-segment that this lane-segment belongs to
        :param e_e_lane_type: Type of lane-segment
        :param e_Cnt_traffic_control_bar_count: Total number of traffic-control-bars in this lane-segment
        :param as_traffic_control_bar: traffic-control-bars in this lane-segment
        :param e_Cnt_left_adjacent_lane_count: Total number of lane-segments to the left of this lane-segment
        :param as_left_adjacent_lanes: Lane-segments to the left of this lane-segment
        :param e_Cnt_right_adjacent_lane_count: Total number of lane-segments to the right of this lane-segment
        :param as_right_adjacent_lanes: Lane-segments to the right of this lane-segment
        :param e_Cnt_downstream_lane_count: Total number of lane-segments downstream of this lane-segment
        :param as_downstream_lanes: Lane-segments downstream of this lane-segment
        :param e_Cnt_upstream_lane_count: Total number of lane-segments upstream of this lane-segment
        :param as_upstream_lanes: Lane-segments upstream of this lane-segment
        :param e_v_nominal_speed: Nominal speed (i.e. speed limit) of this lane-segment
        :param e_l_length: Lane segment length in meters
        :param e_Cnt_num_active_lane_attributes: Number of active lane attributes
        :param a_i_active_lane_attribute_indices: Array that holds the indices of the active lane attributes in a_cmp_lane_attributes
        :param a_cmp_lane_attributes: Array where each element corresponds to a different lane attribute
        :param a_cmp_lane_attribute_confidences: Array where each element corresponds to the confidence in the respective lane attribute assignment
        :param e_Cnt_lane_overlap_count: Total number of lane overlaps for this lane segment
        :param as_lane_overlaps: Lane overlap information for this lane segment
        """
        self.e_i_lane_segment_id = e_i_lane_segment_id
        self.e_i_road_segment_id = e_i_road_segment_id
        self.e_e_lane_type = e_e_lane_type
        self.e_Cnt_traffic_control_bar_count = e_Cnt_traffic_control_bar_count
        self.as_traffic_control_bar = as_traffic_control_bar
        self.e_Cnt_left_adjacent_lane_count = e_Cnt_left_adjacent_lane_count
        self.as_left_adjacent_lanes = as_left_adjacent_lanes
        self.e_Cnt_right_adjacent_lane_count = e_Cnt_right_adjacent_lane_count
        self.as_right_adjacent_lanes = as_right_adjacent_lanes
        self.e_Cnt_downstream_lane_count = e_Cnt_downstream_lane_count
        self.as_downstream_lanes = as_downstream_lanes
        self.e_Cnt_upstream_lane_count = e_Cnt_upstream_lane_count
        self.as_upstream_lanes = as_upstream_lanes
        self.e_v_nominal_speed = e_v_nominal_speed
        self.e_l_length = e_l_length
        self.e_Cnt_num_active_lane_attributes = e_Cnt_num_active_lane_attributes
        self.a_i_active_lane_attribute_indices = a_i_active_lane_attribute_indices
        self.a_cmp_lane_attributes = a_cmp_lane_attributes
        self.a_cmp_lane_attribute_confidences = a_cmp_lane_attribute_confidences
        self.e_Cnt_lane_overlap_count = e_Cnt_lane_overlap_count
        self.as_lane_overlaps = as_lane_overlaps

    def serialize(self) -> TsSYSSceneLaneSegmentBase:
        pubsub_msg = TsSYSSceneLaneSegmentBase()

        pubsub_msg.e_i_lane_segment_id = self.e_i_lane_segment_id
        pubsub_msg.e_i_road_segment_id = self.e_i_road_segment_id
        pubsub_msg.e_e_lane_type = self.e_e_lane_type.value

        pubsub_msg.e_Cnt_traffic_control_bar_count = self.e_Cnt_traffic_control_bar_count
        for i in range(pubsub_msg.e_Cnt_traffic_control_bar_count):
            pubsub_msg.as_traffic_control_bar[i] = self.as_traffic_control_bar[i].serialize()

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

        pubsub_msg.e_l_length = self.e_l_length

        pubsub_msg.e_Cnt_num_active_lane_attributes = self.e_Cnt_num_active_lane_attributes
        pubsub_msg.a_i_active_lane_attribute_indices = self.a_i_active_lane_attribute_indices

        pubsub_msg.a_cmp_lane_attributes = np.array(self.a_cmp_lane_attributes) # These are sparse arrays. So copy the entire array
        pubsub_msg.a_cmp_lane_attribute_confidences = self.a_cmp_lane_attribute_confidences # These are sparse arrays. So copy the entire array

        pubsub_msg.e_Cnt_lane_overlap_count = self.e_Cnt_lane_overlap_count
        for i in range(pubsub_msg.e_Cnt_lane_overlap_count):
            pubsub_msg.as_lane_overlaps[i] = self.as_lane_overlaps[i].serialize()

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSSceneLaneSegmentBase):
        as_traffic_control_bar = list()
        for i in range(pubsubMsg.e_Cnt_traffic_control_bar_count):
            as_traffic_control_bar.append(
                TrafficControlBar.deserialize(pubsubMsg.as_traffic_control_bar[i]))

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

        as_lane_overlaps = list()
        for i in range(pubsubMsg.e_Cnt_lane_overlap_count):
            as_lane_overlaps.append(LaneOverlap.deserialize(pubsubMsg.as_lane_overlaps[i]))

        # Convert Numpy array to list and convert elements to respective enumerations
        lane_attributes = pubsubMsg.a_cmp_lane_attributes.tolist()
        for i in range(SceneLaneSegmentBase.num_lane_attributes):
            lane_attributes[i] = SceneLaneSegmentBase.lane_attribute_types[RoutePlanLaneSegmentAttr(i)](lane_attributes[i])

        return cls(pubsubMsg.e_i_lane_segment_id, pubsubMsg.e_i_road_segment_id, MapLaneType(pubsubMsg.e_e_lane_type),
                   pubsubMsg.e_Cnt_traffic_control_bar_count, as_traffic_control_bar,
                   pubsubMsg.e_Cnt_left_adjacent_lane_count, as_left_adjacent_lanes,
                   pubsubMsg.e_Cnt_right_adjacent_lane_count, as_right_adjacent_lanes,
                   pubsubMsg.e_Cnt_downstream_lane_count, as_downstream_lanes,
                   pubsubMsg.e_Cnt_upstream_lane_count, as_upstream_lanes,
                   pubsubMsg.e_v_nominal_speed,
                   pubsubMsg.e_l_length,
                   pubsubMsg.e_Cnt_num_active_lane_attributes,
                   pubsubMsg.a_i_active_lane_attribute_indices[:pubsubMsg.e_Cnt_num_active_lane_attributes],
                   lane_attributes,
                   pubsubMsg.a_cmp_lane_attribute_confidences[:MAX_LANE_ATTRIBUTES],
                   pubsubMsg.e_Cnt_lane_overlap_count, as_lane_overlaps)


class SceneStaticBase(PUBSUB_MSG_IMPL):
    e_Cnt_num_lane_segments = int
    as_scene_lane_segments = List[SceneLaneSegmentBase]
    e_Cnt_num_road_segments = int
    as_scene_road_segment = List[SceneRoadSegment]
    as_static_traffic_control_device = List[StaticTrafficControlDevice]
    as_dynamic_traffic_control_device = List[DynamicTrafficControlDevice]
    def __init__(self, e_Cnt_num_lane_segments: int, as_scene_lane_segments: List[SceneLaneSegmentBase],
                 e_Cnt_num_road_segments: int, as_scene_road_segment: List[SceneRoadSegment],
                 as_static_traffic_control_device: List[StaticTrafficControlDevice],
                 as_dynamic_traffic_control_device: List[DynamicTrafficControlDevice]):
        """
        Scene provider's static scene information
        :param e_Cnt_num_lane_segments: Total number of lane-segments in the static scene
        :param as_scene_lane_segments: All lane-segments in the static scene
        :param e_Cnt_num_road_segments: Total number of road-segments in the static scene
        :param as_scene_road_segment: All road-segments in the static scene
        :param as_static_traffic_control_device: Static traffic control devices (TCDs) in the scene
        :param as_dynamic_traffic_control_device: Dynamic traffic control devices (TCDs) in the scene
        """
        self.e_Cnt_num_lane_segments = e_Cnt_num_lane_segments
        self.as_scene_lane_segments = as_scene_lane_segments
        self.e_Cnt_num_road_segments = e_Cnt_num_road_segments
        self.as_scene_road_segment = as_scene_road_segment
        self.as_static_traffic_control_device = as_static_traffic_control_device
        self.as_dynamic_traffic_control_device = as_dynamic_traffic_control_device

    def serialize(self) -> TsSYSSceneStaticBase:
        pubsub_msg = TsSYSSceneStaticBase()

        pubsub_msg.e_Cnt_num_lane_segments = self.e_Cnt_num_lane_segments
        for i in range(pubsub_msg.e_Cnt_num_lane_segments):
            pubsub_msg.as_scene_lane_segments[i] = self.as_scene_lane_segments[i].serialize()

        pubsub_msg.e_Cnt_num_road_segments = self.e_Cnt_num_road_segments
        for i in range(pubsub_msg.e_Cnt_num_road_segments):
            pubsub_msg.as_scene_road_segment[i] = self.as_scene_road_segment[i].serialize()

        pubsub_msg.e_Cnt_static_traffic_control_device_count = len(self.as_static_traffic_control_device)
        for i in range(len(self.as_static_traffic_control_device)):
            pubsub_msg.as_static_traffic_control_device[i] = self.as_static_traffic_control_device[i].serialize()

        pubsub_msg.e_Cnt_dynamic_traffic_control_device_count = len(self.as_dynamic_traffic_control_device)
        for i in range(len(self.as_dynamic_traffic_control_device)):
            pubsub_msg.as_dynamic_traffic_control_device[i] = self.as_dynamic_traffic_control_device[i].serialize()

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSSceneStaticBase):

        lane_segments = list()
        for i in range(pubsubMsg.e_Cnt_num_lane_segments):
            lane_segments.append(SceneLaneSegmentBase.deserialize(pubsubMsg.as_scene_lane_segments[i]))

        road_segments = list()
        for i in range(pubsubMsg.e_Cnt_num_road_segments):
            road_segments.append(SceneRoadSegment.deserialize(pubsubMsg.as_scene_road_segment[i]))

        static_traffic_control_device = list()
        for i in range(pubsubMsg.e_Cnt_static_traffic_control_device_count):
            static_traffic_control_device.append(StaticTrafficControlDevice.deserialize(pubsubMsg.as_static_traffic_control_device[i]))

        dynamic_traffic_control_device = list()
        for i in range(pubsubMsg.e_Cnt_dynamic_traffic_control_device_count):
            dynamic_traffic_control_device.append(DynamicTrafficControlDevice.deserialize(pubsubMsg.as_dynamic_traffic_control_device[i]))

        return cls(pubsubMsg.e_Cnt_num_lane_segments,
                   lane_segments,
                   pubsubMsg.e_Cnt_num_road_segments,
                   road_segments,
                   static_traffic_control_device, dynamic_traffic_control_device)


class DataSceneStatic(PUBSUB_MSG_IMPL):
    e_b_Valid = bool
    s_RecvTimestamp = Timestamp
    e_l_perception_horizon_front = float
    e_l_perception_horizon_rear = float
    s_MapOrigin = MapOrigin
    s_SceneStaticBase = SceneStaticBase
    s_SceneStaticGeometry = SceneStaticGeometry
    s_NavigationPlan = NavigationPlan

    def __init__(self, e_b_Valid: bool, s_RecvTimestamp: Timestamp, e_l_perception_horizon_front: float, e_l_perception_horizon_rear: float,
                 s_MapOrigin: MapOrigin, s_SceneStaticBase: SceneStaticBase, s_SceneStaticGeometry: SceneStaticGeometry,
                 s_NavigationPlan: NavigationPlan):
        self.e_b_Valid = e_b_Valid
        self.s_RecvTimestamp = s_RecvTimestamp
        self.e_l_perception_horizon_front = e_l_perception_horizon_front
        self.e_l_perception_horizon_rear = e_l_perception_horizon_rear
        self.s_MapOrigin = s_MapOrigin
        self.s_SceneStaticBase = s_SceneStaticBase
        self.s_SceneStaticGeometry = s_SceneStaticGeometry
        self.s_NavigationPlan = s_NavigationPlan

    def serialize(self) -> TsSYSDataSceneStatic:
        pubsub_msg = TsSYSDataSceneStatic()

        pubsub_msg.e_b_Valid = self.e_b_Valid
        pubsub_msg.s_RecvTimestamp = self.s_RecvTimestamp.serialize()
        pubsub_msg.e_l_perception_horizon_front = self.e_l_perception_horizon_front
        pubsub_msg.e_l_perception_horizon_rear = self.e_l_perception_horizon_rear
        pubsub_msg.s_MapOrigin = self.s_MapOrigin.serialize()
        pubsub_msg.s_SceneStaticBase = self.s_SceneStaticBase.serialize()
        pubsub_msg.s_SceneStaticGeometry = self.s_SceneStaticGeometry.serialize()
        pubsub_msg.s_NavigationPlan = self.s_NavigationPlan.serialize()

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSDataSceneStatic):
        return cls(pubsubMsg.e_b_Valid,
                   Timestamp.deserialize(pubsubMsg.s_RecvTimestamp),
                   pubsubMsg.e_l_perception_horizon_front,
                   pubsubMsg.e_l_perception_horizon_rear,
                   MapOrigin.deserialize(pubsubMsg.s_MapOrigin),
                   SceneStaticBase.deserialize(pubsubMsg.s_SceneStaticBase),
                   SceneStaticGeometry.deserialize(pubsubMsg.s_SceneStaticGeometry),
                   NavigationPlan.deserialize(pubsubMsg.s_NavigationPlan))


class SceneStatic(PUBSUB_MSG_IMPL):
    s_Header = Header
    s_Data = DataSceneStatic

    def __init__(self, s_Header: Header, s_Data: DataSceneStatic):
        self.s_Header = s_Header
        self.s_Data = s_Data

    def serialize(self) -> TsSYSSceneStatic:
        pubsub_msg = TsSYSSceneStatic()

        pubsub_msg.s_Header = self.s_Header.serialize()
        pubsub_msg.s_Data = self.s_Data.serialize()

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSSceneStatic):
        return cls(Header.deserialize(pubsubMsg.s_Header),
                   DataSceneStatic.deserialize(pubsubMsg.s_Data))
