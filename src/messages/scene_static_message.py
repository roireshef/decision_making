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

MAX_NOMINAL_PATH_POINT_FIELDS = 10


class MapLaneType(Enum):
    ControlledAccess_DividedRoadLane = 0
    ControlledAccess_DividedRoadInterchangeLinks = 1
    NonControlledAccess_DividedRoadLane = 2
    ControlledAccess_NonDividedRoadLane = 3
    AccessRampLane = 4
    NonDividedRoadLane_NonControlledAccess = 5
    LocalRoadLane = 6
    Right_RoadEdgeLane_EmergencyLane = 7
    Left_RoadEdgeLane_EmergencyLane = 8
    TurnLane = 9
    HOVLane = 10
    BidirectionalLane = 11


class MapRoadSegmentType(Enum):
    Normal = 0
    Intersection = 1
    TurnOnly = 2
    Unknown = 3


class MovingDirection(Enum):
    Adjacent_or_same_dir = 1
    Opposing = 2
    Centerlane = 3


class ManeuverType(Enum):
    STRAIGHT_CONNECTION = 0
    LEFT_TURN_CONNECTION = 1
    RIGHT_TURN_CONNECTION = 2
    LEFT_MERGE_CONNECTION = 3
    RIGHT_MERGE_CONNECTION = 4
    LEFT_EXIT_CONNECTION = 5
    RIGHT_EXIT_CONNECTION = 6
    LEFT_FORK_CONNECTION = 7
    RIGHT_FORK_CONNECTION = 8
    TRAFFIC_CIRCLE_CONNECTION = 9
    LEFT_LANE_CHANGE_CONNECTION = 10
    RIGHT_LANE_CHANGE_CONNECTION = 11
    NO_ALLOWED_CONNECTION = 12


class MapLaneMarkerType(Enum):
    MapLaneMarkerType_None = 0
    MapLaneMarkerType_ThinDashed_SingleLine = 1
    MapLaneMarkerType_ThickDashed_SingleLine = 2
    MapLaneMarkerType_ThinSolid_SingleLine = 3
    MapLaneMarkerType_ThickSolid_SingleLine = 4
    MapLaneMarkerType_ThinDashed_DoubleLine = 5
    MapLaneMarkerType_ThickDashed_DoubleLine = 6
    MapLaneMarkerType_ThinSolid_DoubleLine = 7
    MapLaneMarkerType_ThickSolid_DoubleLine = 8
    MapLaneMarkerType_ThinDoubleLeftSolid_RightDashedLine = 9
    MapLaneMarkerType_ThickDoubleLefSolid_RightDashedLine = 10
    MapLaneMarkerType_ThinDoubleLeftDashed_RightSolidLine = 11
    MapLaneMarkerType_ThickDoubleLeftDashed_RightSolidLine = 12
    MapLaneMarkerType_ThinDashed_TripleLine = 13
    MapLaneMarkerType_ThickDashed_TripleLine = 14
    MapLaneMarkerType_ThinSolid_TripleLine = 15
    MapLaneMarkerType_ThickSolid_TripleLine = 16
    MapLaneMarkerType_DashedSingleLine_BottsDots = 17
    MapLaneMarkerType_SolidSingleLine_BottsDots = 18
    MapLaneMarkerType_DashedDoubleLine_BottsDots = 19
    MapLaneMarkerType_SolidDoubleLine_BottsDots = 20
    MapLaneMarkerType_DoubleLeftSolid_RightDashedLine_BottsDots = 21
    MapLaneMarkerType_DoubleLeftDashed_RightSolidLine_BottsDots = 22
    MapLaneMarkerType_DashedTripleLine_BottsDots = 23
    MapLaneMarkerType_SolidTripleLine_BottsDots = 24
    MapLaneMarkerType_VirtualInferredLine = 25
    MapLaneMarkerType_Unknown = 255


class RoadObjectType(Enum):
    Yield = 0
    StopSign = 1
    VerticalStack_RYG_TrafficLight = 2
    HorizontalStack_RYG_TrafficLight = 3
    YieldVerticalStack_RYG_TrafficLight_WithInd = 4
    HorizontalStack_RYG_TrafficLight_WithInd = 5
    Red_TrafficLight = 6
    Yellow_TrafficLight = 7
    StopBar_Left = 8
    StopBar_Right = 9
    LaneLeftTurnMarker_Start = 10
    LaneLeftTurnMarker_End = 11
    LaneRightTurnMarker_Start = 12
    LaneRightTurnMarker_End = 13
    LaneStraightLeftTurnMarker_Start = 14
    LaneStraightLeftTurnMarker_End = 15
    LaneStraightRightTurnMarker_Start = 16
    LaneStraightRightTurnMarker_End = 17
    LaneStopMarker_MidLeft = 18
    LaneStopMarker_MidRight = 19
    LaneStopMarker_Diamond_Start = 20
    LaneStopMarker_Diamond_end = 21
    LightPost = 22
    Dashed_LaneMarker_MidStart = 23
    Dashed_LaneMarker_MidEnd = 24


class TrafficSignalState(Enum):
    NO_DETECTION = 0
    RED = 1
    YELLOW = 2
    RED_YELLOW = 3
    GREEN = 4
    GREEN_YELLOW = 6
    FLASHING_BIT = 8  # (flashing modifier, not valid by itself)
    FLASHING_RED = 9
    FLASHING_YELLOW = 10
    FLASHING_GREEN = 12
    UP_DOWN_ARROW_BIT = 16  # (up/down arrow modifier, not valid by itself)
    STRAIGHT_ARROW_RED = 17
    STRAIGHT_ARROW_YELLOW = 18
    STRAIGHT_ARROW_GREEN = 20
    LEFT_ARROW_BIT = 32  # (left arrow modifier, not valid by itself)
    LEFT_ARROW_RED = 33
    LEFT_ARROW_YELLOW = 34
    LEFT_ARROW_GREEN = 36
    FLASHING_LEFT_ARROW_RED = 41
    FLASHING_LEFT_ARROW_YELLOW = 42
    FLASHING_LEFT_ARROW_GREEN = 44
    RIGHT_ARROW_BIT = 64  # (right arrow modifier, not valid by itself)
    RIGHT_ARROW_RED = 65
    RIGHT_ARROW_YELLOW = 66
    RIGHT_ARROW_GREEN = 68
    FLASHING_RIGHT_ARROW_RED = 73
    FLASHING_RIGHT_ARROW_YELLOW = 74
    FLASHING_RIGHT_ARROW_GREEN = 76
    RAILROAD_CROSSING_UNKNOWN = 96
    RAILROAD_CROSSING_RED = 97
    RAILROAD_CROSSING_CLEAR = 100
    RAILROAD_CROSSING_BLOCKED = 104
    PEDISTRIAN_SAME_DIR_UNKNOWN = 108
    PEDISTRIAN_SAME_DIR_STOPPED = 109
    PEDISTRIAN_SAME_DIR_WARNING = 110
    PEDISTRIAN_SAME_DIR_CROSS = 112
    PEDISTRIAN_PERP_DIR_UNKNOWN = 116
    PEDISTRIAN_PERP_DIR_STOPPED = 117
    PEDISTRIAN_PERP_DIR_WARNING = 118
    PEDISTRIAN_PERP_DIR_CROSS = 120
    UNKNOWN = 121
    OFF = 122


class NominalPathPoint(Enum):
    CeSYS_NominalPathPoint_e_l_EastX = 0
    CeSYS_NominalPathPoint_e_l_NorthY = 1
    CeSYS_NominalPathPoint_e_phi_heading = 2
    CeSYS_NominalPathPoint_e_il_curvature = 3
    CeSYS_NominalPathPoint_e_il2_curvature_rate = 4
    CeSYS_NominalPathPoint_e_phi_cross_slope = 5
    CeSYS_NominalPathPoint_e_phi_along_slope = 6
    CeSYS_NominalPathPoint_e_l_s = 7
    CeSYS_NominalPathPoint_e_l_left_offset = 8
    CeSYS_NominalPathPoint_e_l_right_offset = 9


class SceneRoadSegment(PUBSUB_MSG_IMPL):
    e_Cnt_road_segment_id = int
    e_Cnt_road_id = int
    e_Cnt_lane_segment_id_count = int
    a_Cnt_lane_segment_id = np.ndarray
    e_e_road_segment_type = MapRoadSegmentType
    e_Cnt_upstream_segment_count = int
    a_Cnt_upstream_road_segment_id = np.ndarray
    e_Cnt_downstream_segment_count = int
    a_Cnt_downstream_road_segment_id = np.ndarray

    def __init__(self, e_Cnt_road_segment_id: int, e_Cnt_road_id: int, e_Cnt_lane_segment_id_count: int,
                 a_Cnt_lane_segment_id: np.ndarray, e_e_road_segment_type: MapRoadSegmentType,
                 e_Cnt_upstream_segment_count: int, a_Cnt_upstream_road_segment_id: np.ndarray,
                 e_Cnt_downstream_segment_count: int, a_Cnt_downstream_road_segment_id: np.ndarray) -> None:
        """
        Road-segment information
        :param e_Cnt_road_segment_id: ID of this Road-segment
        :param e_Cnt_road_id: Not relevant for M0
        :param e_Cnt_lane_segment_id_count: Total number of all lane-segments contained within this road-segment
        :param a_Cnt_lane_segment_id: Lane-segments contained within this road-segment
        :param e_e_road_segment_type:
        :param e_Cnt_upstream_segment_count: Total number of upstream road-segments from this road-segment
        :param a_Cnt_upstream_road_segment_id: Upstream road-segments from this road-segment
        :param e_Cnt_downstream_segment_count: Total number of downstream road-segments from this road-segment
        :param a_Cnt_downstream_road_segment_id: Downstream road-segments from this road-segment
        """
        self.e_Cnt_road_segment_id = e_Cnt_road_segment_id
        self.e_Cnt_road_id = e_Cnt_road_id
        self.e_Cnt_lane_segment_id_count = e_Cnt_lane_segment_id_count
        self.a_Cnt_lane_segment_id = a_Cnt_lane_segment_id
        self.e_e_road_segment_type = e_e_road_segment_type
        self.e_Cnt_upstream_segment_count = e_Cnt_upstream_segment_count
        self.a_Cnt_upstream_road_segment_id = a_Cnt_upstream_road_segment_id
        self.e_Cnt_downstream_segment_count = e_Cnt_downstream_segment_count
        self.a_Cnt_downstream_road_segment_id = a_Cnt_downstream_road_segment_id

    def serialize(self) -> TsSYSSceneRoadSegment:
        pubsub_msg = TsSYSSceneRoadSegment()

        pubsub_msg.e_Cnt_road_segment_id = self.e_Cnt_road_segment_id
        pubsub_msg.e_Cnt_road_id = self.e_Cnt_road_id

        pubsub_msg.e_Cnt_lane_segment_id_count = self.e_Cnt_lane_segment_id_count
        pubsub_msg.a_Cnt_lane_segment_id = self.a_Cnt_lane_segment_id

        pubsub_msg.e_e_road_segment_type = self.e_e_road_segment_type.value

        pubsub_msg.e_Cnt_upstream_segment_count = self.e_Cnt_upstream_segment_count
        pubsub_msg.a_Cnt_upstream_road_segment_id = self.a_Cnt_upstream_road_segment_id

        pubsub_msg.e_Cnt_downstream_segment_count = self.e_Cnt_downstream_segment_count
        pubsub_msg.a_Cnt_downstream_road_segment_id = self.a_Cnt_downstream_road_segment_id

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSSceneRoadSegment):
        return cls(pubsubMsg.e_Cnt_road_segment_id,
                   pubsubMsg.e_Cnt_road_id,
                   pubsubMsg.e_Cnt_lane_segment_id_count,
                   pubsubMsg.a_Cnt_lane_segment_id[:pubsubMsg.e_Cnt_lane_segment_id_count],
                   MapRoadSegmentType(pubsubMsg.e_e_road_segment_type),
                   pubsubMsg.e_Cnt_upstream_segment_count,
                   pubsubMsg.a_Cnt_upstream_road_segment_id[:pubsubMsg.e_Cnt_upstream_segment_count],
                   pubsubMsg.e_Cnt_downstream_segment_count,
                   pubsubMsg.a_Cnt_downstream_road_segment_id[:pubsubMsg.e_Cnt_downstream_segment_count])


class SceneRoadIntersection(PUBSUB_MSG_IMPL):
    e_i_road_intersection_id = int
    e_Cnt_lane_coupling_count = int
    a_i_lane_coupling_segment_ids = np.ndarray
    e_Cnt_intersection_road_segment_count = int
    a_i_intersection_road_segment_ids = np.ndarray

    def __init__(self, e_i_road_intersection_id: int,
                 e_Cnt_lane_coupling_count: int, a_i_lane_coupling_segment_ids: np.ndarray,
                 e_Cnt_intersection_road_segment_count: int, a_i_intersection_road_segment_ids: np.ndarray) -> None:
        """
        Road-intersection information
        :param e_i_road_intersection_id: ID of this road-intersection
        :param e_Cnt_lane_coupling_count: Total number of lane-couplings inside this road-intersection
        :param a_i_lane_coupling_segment_ids: Lane-couplings inside this road-intersection: all lane-couplings inside
        a road-intersection are non-virtual, and are equivalent to lane-segments.
        :param e_Cnt_intersection_road_segment_count: Total number of road-segments inside this road-intersection
        :param a_i_intersection_road_segment_ids: Road-segments inside this road-intersection. A road-intersection
        contains all the road-segments going through it, e.g. in a 4-way intersection, the road-intersection will
        contain both the North/South and East/West road-segments.
        """
        self.e_i_road_intersection_id = e_i_road_intersection_id
        self.e_Cnt_lane_coupling_count = e_Cnt_lane_coupling_count
        self.a_i_lane_coupling_segment_ids = a_i_lane_coupling_segment_ids
        self.e_Cnt_intersection_road_segment_count = e_Cnt_intersection_road_segment_count
        self.a_i_intersection_road_segment_ids = a_i_intersection_road_segment_ids

    def serialize(self) -> TsSYSSceneRoadIntersection:
        pubsub_msg = TsSYSSceneRoadIntersection()

        pubsub_msg.e_i_road_intersection_id = self.e_i_road_intersection_id

        pubsub_msg.e_Cnt_lane_coupling_count = self.e_Cnt_lane_coupling_count
        pubsub_msg.a_i_lane_coupling_segment_ids = self.a_i_lane_coupling_segment_ids

        pubsub_msg.e_Cnt_intersection_road_segment_count = self.e_Cnt_intersection_road_segment_count
        pubsub_msg.a_i_intersection_road_segment_ids = self.a_i_intersection_road_segment_ids

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSSceneRoadIntersection):
        return cls(pubsubMsg.e_i_road_intersection_id,
                   pubsubMsg.e_Cnt_lane_coupling_count,
                   pubsubMsg.a_i_lane_coupling_segment_ids[:pubsubMsg.e_Cnt_lane_coupling_count],
                   pubsubMsg.e_Cnt_intersection_road_segment_count,
                   pubsubMsg.a_i_intersection_road_segment_ids[:pubsubMsg.e_Cnt_intersection_road_segment_count])


class AdjacentLane(PUBSUB_MSG_IMPL):
    e_Cnt_lane_segment_id = int
    e_e_moving_direction = MovingDirection
    e_e_lane_type = MapLaneType

    def __init__(self, e_Cnt_lane_segment_id: int, e_e_moving_direction: MovingDirection, e_e_lane_type: MapLaneType):
        self.e_Cnt_lane_segment_id = e_Cnt_lane_segment_id
        self.e_e_moving_direction = e_e_moving_direction
        self.e_e_lane_type = e_e_lane_type

    def serialize(self) -> TsSYSAdjacentLane:
        pubsub_msg = TsSYSAdjacentLane()

        pubsub_msg.e_Cnt_lane_segment_id = self.e_Cnt_lane_segment_id
        pubsub_msg.e_e_moving_direction = self.e_e_moving_direction.value
        pubsub_msg.e_e_lane_type = self.e_e_lane_type.value

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSAdjacentLane):
        # TODO: hack!! moving direction and map lane type is received badly from map_services, remove when fixed
        return cls(pubsubMsg.e_Cnt_lane_segment_id, MovingDirection(1),
                   MapLaneType(5))


class LaneSegmentConnectivity(PUBSUB_MSG_IMPL):
    e_Cnt_lane_segment_id = int
    e_e_maneuver_type = ManeuverType

    def __init__(self, e_Cnt_lane_segment_id: int, e_e_maneuver_type: ManeuverType):
        self.e_Cnt_lane_segment_id = e_Cnt_lane_segment_id
        self.e_e_maneuver_type = e_e_maneuver_type

    def serialize(self) -> TsSYSLaneSegmentConnectivity:
        pubsub_msg = TsSYSLaneSegmentConnectivity()

        pubsub_msg.e_Cnt_lane_segment_id = self.e_Cnt_lane_segment_id
        pubsub_msg.e_e_maneuver_type = self.e_e_maneuver_type.value

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSLaneSegmentConnectivity):
        return cls(pubsubMsg.e_Cnt_lane_segment_id, ManeuverType(pubsubMsg.e_e_maneuver_type))


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

    # TODO: remove hack of MapLaneMarkerType after SP fix
    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSBoundaryPoint):
        return cls(MapLaneMarkerType(5), pubsubMsg.e_l_s_start, pubsubMsg.e_l_s_end)


class LaneCoupling(PUBSUB_MSG_IMPL):
    e_i_lane_segment_id = int
    e_i_road_intersection_id = int
    e_i_downstream_lane_segment_id = int
    e_i_upstream_lane_segment_id = int
    e_e_maneuver_type = ManeuverType

    def __init__(self, e_i_lane_segment_id: int, e_i_road_intersection_id: int, e_i_downstream_lane_segment_id: int,
                 e_i_upstream_lane_segment_id: int, e_e_maneuver_type: ManeuverType):
        """
        Lane-coupling information. Generally, a lane-coupling connects two lane-segments with the same moving direction.
        'Virtual' lane-couplings (e.g. between two lane-segments on a straight road) have ID's of 0.
        Non-virtual lane-couplings (e.g. inside a road-intersection) are equivalent to lane-segments and have non-zero ID's.
        :param e_i_lane_segment_id: ID of this lane-coupling/lane-segment. ID=0 if this is a virtual lane-coupling
        :param e_i_road_intersection_id: ID of the road-intersection that this lane-coupling is in. ID=0 if N/A.
        :param e_i_downstream_lane_segment_id: ID of the lane-segment downstream from this lane-coupling
        :param e_i_upstream_lane_segment_id: ID of the lane-segment upstream from this lane-coupling
        :param e_e_maneuver_type:
        """
        self.e_i_lane_segment_id = e_i_lane_segment_id
        self.e_i_road_intersection_id = e_i_road_intersection_id
        self.e_i_downstream_lane_segment_id = e_i_downstream_lane_segment_id
        self.e_i_upstream_lane_segment_id = e_i_upstream_lane_segment_id
        self.e_e_maneuver_type = e_e_maneuver_type

    def serialize(self) -> TsSYSLaneCoupling:
        pubsub_msg = TsSYSLaneCoupling()

        pubsub_msg.e_i_lane_segment_id = self.e_i_lane_segment_id
        pubsub_msg.e_i_road_intersection_id = self.e_i_road_intersection_id
        pubsub_msg.e_i_downstream_lane_segment_id = self.e_i_downstream_lane_segment_id
        pubsub_msg.e_i_upstream_lane_segment_id = self.e_i_upstream_lane_segment_id
        pubsub_msg.e_e_maneuver_type = self.e_e_maneuver_type.value

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSLaneCoupling):
        return cls(pubsubMsg.e_i_lane_segment_id, pubsubMsg.e_i_road_intersection_id,
                   pubsubMsg.e_i_downstream_lane_segment_id, pubsubMsg.e_i_upstream_lane_segment_id,
                   ManeuverType(pubsubMsg.e_e_maneuver_type))


class StaticTrafficFlowControl(PUBSUB_MSG_IMPL):
    e_e_road_object_type = RoadObjectType
    e_l_station = float
    e_Pct_confidence = float

    def __init__(self, e_e_road_object_type: RoadObjectType, e_l_station: float, e_Pct_confidence: float):
        """
        Static traffic-flow-control device, eg. Stop Signs (not relevant for M0)
        :param e_e_road_object_type:
        :param e_l_station:
        :param e_Pct_confidence:
        """
        self.e_e_road_object_type = e_e_road_object_type
        self.e_l_station = e_l_station
        self.e_Pct_confidence = e_Pct_confidence

    def serialize(self) -> TsSYSStaticTrafficFlowControl:
        pubsub_msg = TsSYSStaticTrafficFlowControl()

        pubsub_msg.e_e_road_object_type = self.e_e_road_object_type.value
        pubsub_msg.e_l_station = self.e_l_station
        pubsub_msg.e_Pct_confidence = self.e_Pct_confidence

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSStaticTrafficFlowControl):
        return cls(RoadObjectType(pubsubMsg.e_e_road_object_type), pubsubMsg.e_l_station,
                   pubsubMsg.e_Pct_confidence)


class DynamicStatus(PUBSUB_MSG_IMPL):
    e_e_status = TrafficSignalState
    e_Pct_confidence = float

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


class DynamicTrafficFlowControl(PUBSUB_MSG_IMPL):
    e_e_road_object_type = RoadObjectType
    e_l_station = float
    e_Cnt_dynamic_status_count = int
    as_dynamic_status = List[DynamicStatus]

    def __init__(self, e_e_road_object_type: RoadObjectType, e_l_station: float,
                 e_Cnt_dynamic_status_count: int, as_dynamic_status: List[DynamicStatus]):
        """
        Dynamic traffic-flow-control device, e.g. Traffic lights (not relevant for M0)
        :param e_e_road_object_type:
        :param e_l_station:
        :param e_Cnt_dynamic_status_count:
        :param as_dynamic_status:
        """
        self.e_e_road_object_type = e_e_road_object_type
        self.e_l_station = e_l_station
        self.e_Cnt_dynamic_status_count = e_Cnt_dynamic_status_count
        self.as_dynamic_status = as_dynamic_status

    def serialize(self) -> TsSYSDynamicTrafficFlowControl:
        pubsub_msg = TsSYSDynamicTrafficFlowControl()

        pubsub_msg.e_e_road_object_type = self.e_e_road_object_type.value
        pubsub_msg.e_l_station = self.e_l_station
        pubsub_msg.e_Cnt_dynamic_status_count = self.e_Cnt_dynamic_status_count

        for i in range(pubsub_msg.e_Cnt_dynamic_status_count):
            pubsub_msg.as_dynamic_status[i] = self.as_dynamic_status[i].serialize()

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSDynamicTrafficFlowControl):
        dynamic_statuses = list()
        for i in range(pubsubMsg.e_Cnt_dynamic_status_count):
            dynamic_statuses.append(DynamicStatus.deserialize(pubsubMsg.as_dynamic_status[i]))
        return cls(RoadObjectType(pubsubMsg.e_e_road_object_type), pubsubMsg.e_l_station,
                   pubsubMsg.e_Cnt_dynamic_status_count,
                   dynamic_statuses)


class SceneLaneSegment(PUBSUB_MSG_IMPL):
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
    e_Cnt_nominal_path_point_count = int
    a_nominal_path_points = np.ndarray
    e_Cnt_left_boundary_points_count = int
    as_left_boundary_points = List[BoundaryPoint]
    e_Cnt_right_boundary_points_count = int
    as_right_boundary_points = List[BoundaryPoint]
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
                 e_v_nominal_speed: float, e_Cnt_nominal_path_point_count: int, a_nominal_path_points: np.ndarray,
                 e_Cnt_left_boundary_points_count: int, as_left_boundary_points: List[BoundaryPoint],
                 e_Cnt_right_boundary_points_count: int, as_right_boundary_points: List[BoundaryPoint],
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
        :param e_Cnt_nominal_path_point_count: Total number of points that specify the nominal-path (i.e. center of lane) for this lane-segment
        :param a_nominal_path_points: Points that specify the nominal-path (i.e. center of lane) for this lane-segment.
               Its shape has to be [e_Cnt_nominal_path_point_count X MAX_NOMINAL_PATH_POINT_FIELDS].
        :param e_Cnt_left_boundary_points_count: Total number of points that specify the left-boundary for this lane-segment
        :param as_left_boundary_points: Points that specify the left-boundary for this lane-segment
        :param e_Cnt_right_boundary_points_count: Total number of points that specify the right-boundary for this lane-segment
        :param as_right_boundary_points: Points that specify the right-boundary for this lane-segment
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
        self.e_Cnt_nominal_path_point_count = e_Cnt_nominal_path_point_count
        self.a_nominal_path_points = a_nominal_path_points
        self.e_Cnt_left_boundary_points_count = e_Cnt_left_boundary_points_count
        self.as_left_boundary_points = as_left_boundary_points
        self.e_Cnt_right_boundary_points_count = e_Cnt_right_boundary_points_count
        self.as_right_boundary_points = as_right_boundary_points
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

        pubsub_msg.e_Cnt_nominal_path_point_count = self.e_Cnt_nominal_path_point_count
        pubsub_msg.a_nominal_path_points = self.a_nominal_path_points

        pubsub_msg.e_Cnt_left_boundary_points_count = self.e_Cnt_left_boundary_points_count
        for i in range(pubsub_msg.e_Cnt_left_boundary_points_count):
            pubsub_msg.as_left_boundary_points[i] = self.as_left_boundary_points[i].serialize()

        pubsub_msg.e_Cnt_right_boundary_points_count = self.e_Cnt_right_boundary_points_count
        for i in range(pubsub_msg.e_Cnt_right_boundary_points_count):
            pubsub_msg.as_right_boundary_points[i] = self.as_right_boundary_points[i].serialize()

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
        # TODO: counter values are invalid??
        # TODO: data dict is messed up??
        # if pubsubMsg.e_Cnt_downstream_lane_count > 0:
        #     [as_downstream_lanes.append(LaneSegmentConnectivity.deserialize(ds_lane))
        #      for ds_lane in pubsubMsg.as_downstream_lanes if ds_lane.e_Cnt_lane_segment_id > 0]
        for i in range(min(1,pubsubMsg.e_Cnt_downstream_lane_count)):
            as_downstream_lanes.append(LaneSegmentConnectivity.deserialize(pubsubMsg.as_downstream_lanes[i]))

        as_upstream_lanes = list()
        for i in range(pubsubMsg.e_Cnt_upstream_lane_count):
            as_upstream_lanes.append(LaneSegmentConnectivity.deserialize(pubsubMsg.as_upstream_lanes[i]))

        a_nominal_path_points = pubsubMsg.a_nominal_path_points[:pubsubMsg.e_Cnt_nominal_path_point_count,
                                :MAX_NOMINAL_PATH_POINT_FIELDS]

        as_left_boundary_points = list()
        for i in range(pubsubMsg.e_Cnt_left_boundary_points_count):
            as_left_boundary_points.append(BoundaryPoint.deserialize(pubsubMsg.as_left_boundary_points[i]))

        as_right_boundary_points = list()
        for i in range(pubsubMsg.e_Cnt_right_boundary_points_count):
            as_right_boundary_points.append(BoundaryPoint.deserialize(pubsubMsg.as_right_boundary_points[i]))

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
                   pubsubMsg.e_Cnt_nominal_path_point_count, a_nominal_path_points,
                   pubsubMsg.e_Cnt_left_boundary_points_count, as_left_boundary_points,
                   pubsubMsg.e_Cnt_right_boundary_points_count, as_right_boundary_points,
                   pubsubMsg.e_i_downstream_road_intersection_id,
                   pubsubMsg.e_Cnt_lane_coupling_count, as_lane_coupling)


class DataSceneStatic(PUBSUB_MSG_IMPL):
    e_b_Valid = bool
    s_RecvTimestamp = Timestamp
    s_ComputeTimestamp = Timestamp
    e_l_perception_horizon_front = float
    e_l_perception_horizon_rear = float
    e_Cnt_num_lane_segments = int
    as_scene_lane_segment = List[SceneLaneSegment]
    e_Cnt_num_road_intersections = int
    as_scene_road_intersection = List[SceneRoadIntersection]
    e_Cnt_num_road_segments = int
    as_scene_road_segment = List[SceneRoadSegment]

    def __init__(self, e_b_Valid: bool, s_RecvTimestamp:Timestamp, s_ComputeTimestamp: Timestamp, e_l_perception_horizon_front: float,
                 e_l_perception_horizon_rear: float,
                 e_Cnt_num_lane_segments: int, as_scene_lane_segment: List[SceneLaneSegment],
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

    def serialize(self) -> TsSYSDataSceneStatic:
        pubsub_msg = TsSYSDataSceneStatic()

        pubsub_msg.e_b_Valid = self.e_b_Valid
        pubsub_msg.s_RecvTimestamp = self.s_RecvTimestamp.serialize()
        pubsub_msg.s_ComputeTimestamp = self.s_ComputeTimestamp.serialize()
        pubsub_msg.e_l_perception_horizon_front = self.e_l_perception_horizon_front
        pubsub_msg.e_l_perception_horizon_rear = self.e_l_perception_horizon_rear

        pubsub_msg.e_Cnt_num_lane_segments = self.e_Cnt_num_lane_segments
        for i in range(pubsub_msg.e_Cnt_num_lane_segments):
            pubsub_msg.as_scene_lane_segment[i] = self.as_scene_lane_segment[i].serialize()

        pubsub_msg.e_Cnt_num_road_intersections = self.e_Cnt_num_road_intersections
        for i in range(pubsub_msg.e_Cnt_num_road_intersections):
            pubsub_msg.as_scene_road_intersection[i] = self.as_scene_road_intersection[i].serialize()

        pubsub_msg.e_Cnt_num_road_segments = self.e_Cnt_num_road_segments
        for i in range(pubsub_msg.e_Cnt_num_road_segments):
            pubsub_msg.as_scene_road_segment[i] = self.as_scene_road_segment[i].serialize()

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSDataSceneStatic):

        lane_segments = list()
        for i in range(pubsubMsg.e_Cnt_num_lane_segments):
            lane_segments.append(SceneLaneSegment.deserialize(pubsubMsg.as_scene_lane_segment[i]))

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


class SceneStatic(PUBSUB_MSG_IMPL):
    s_Header = Header
    s_MapOrigin = MapOrigin
    s_Data = DataSceneStatic

    def __init__(self, s_Header: Header, s_MapOrigin: MapOrigin, s_Data: DataSceneStatic):
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
                   DataSceneStatic.deserialize(pubsubMsg.s_Data))
