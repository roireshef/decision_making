from enum import Enum
from typing import List

from common_data.interface.py.idl_generated_files.Rte_Types.sub_structures.TsSYS_DataSceneStatic import \
    TsSYSDataSceneStatic
from common_data.lcm.generatedFiles.gm_lcm import LcmNumpyArray
from numpy import np

from Rte_Types.sub_structures import TsSYSAdjacentLane, TsSYSLaneManeuver, TsSYSBoundaryPoint, TsSYSLaneCoupling, \
    TsSYSNominalPathPoint, TsSYSStaticTrafficFlowControl, TsSYSDynamicStatus, TsSYSDynamicTrafficFlowControl, \
    TsSYSSceneLaneSegment
from common_data.interface.py.idl_generated_files.Rte_Types.TsSYS_SceneStatic import TsSYSSceneStatic
from common_data.interface.py.idl_generated_files.Rte_Types.sub_structures.TsSYS_SceneRoadIntersection import \
    TsSYSSceneRoadIntersection
from common_data.interface.py.idl_generated_files.Rte_Types.sub_structures.TsSYS_SceneRoadSegment import \
    TsSYSSceneRoadSegment
from decision_making.src.global_constants import PUBSUB_MSG_IMPL
from decision_making.src.messages.scene_dynamic_message import Timestamp, MapOrigin, Header


# MAX_SCENE_LANE_SEGMENTS = 128
# MAX_SCENE_ROAD_INTERSECTIONS = 64
# MAX_SCENE_ROAD_SEGMENTS = 64

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
   e_MapRoadType_Normal = 0,
   e_MapRoadType_Intersection = 1,
   e_MapRoadType_TurnOnly = 2,
   e_MapRoadType_Unknown = 3

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

    MapLaneMarkerType_None                                         = 0
    MapLaneMarkerType_ThinDashed_SingleLine                        = 1
    MapLaneMarkerType_ThickDashed_SingleLine                       = 2
    MapLaneMarkerType_ThinSolid_SingleLine                         = 3
    MapLaneMarkerType_ThickSolid_SingleLine                        = 4
    MapLaneMarkerType_ThinDashed_DoubleLine                        = 5
    MapLaneMarkerType_ThickDashed_DoubleLine                       = 6
    MapLaneMarkerType_ThinSolid_DoubleLine                         = 7
    MapLaneMarkerType_ThickSolid_DoubleLine                        = 8
    MapLaneMarkerType_ThinDoubleLeftSolid_RightDashedLine          = 9
    MapLaneMarkerType_ThickDoubleLefSolid_RightDashedLine          = 10
    MapLaneMarkerType_ThinDoubleLeftDashed_RightSolidLine          = 11
    MapLaneMarkerType_ThickDoubleLeftDashed_RightSolidLine         = 12
    MapLaneMarkerType_ThinDashed_TripleLine                        = 13
    MapLaneMarkerType_ThickDashed_TripleLine                       = 14
    MapLaneMarkerType_ThinSolid_TripleLine                         = 15
    MapLaneMarkerType_ThickSolid_TripleLine                        = 16
    MapLaneMarkerType_DashedSingleLine_BottsDots                   = 17
    MapLaneMarkerType_SolidSingleLine_BottsDots                    = 18
    MapLaneMarkerType_DashedDoubleLine_BottsDots                   = 19
    MapLaneMarkerType_SolidDoubleLine_BottsDots                    = 20
    MapLaneMarkerType_DoubleLeftSolid_RightDashedLine_BottsDots    = 21
    MapLaneMarkerType_DoubleLeftDashed_RightSolidLine_BottsDots    = 22
    MapLaneMarkerType_DashedTripleLine_BottsDots                   = 23
    MapLaneMarkerType_SolidTripleLine_BottsDots                    = 24
    MapLaneMarkerType_VirtualInferredLine                          = 25
    MapLaneMarkerType_Unknown                                      = 255


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


class RoadObjectType(Enum):

    Yield                                               =  0
    StopSign                                            =  1
    VerticalStack_RYG_TrafficLight                      =  2
    HorizontalStack_RYG_TrafficLight                    =  3
    YieldVerticalStack_RYG_TrafficLight_WithInd         =  4
    HorizontalStack_RYG_TrafficLight_WithInd            =  5
    Red_TrafficLight                                    =  6
    Yellow_TrafficLight                                 =  7
    StopBar_Left                                        =  8
    StopBar_Right                                       =  9
    LaneLeftTurnMarker_Start                            =  10
    LaneLeftTurnMarker_End                              =  11
    LaneRightTurnMarker_Start                           =  12
    LaneRightTurnMarker_End                             =  13
    LaneStraightLeftTurnMarker_Start                    =  14
    LaneStraightLeftTurnMarker_End                      =  15
    LaneStraightRightTurnMarker_Start                   =  16
    LaneStraightRightTurnMarker_End                     =  17
    LaneStopMarker_MidLeft                              =  18
    LaneStopMarker_MidRight                             =  19
    LaneStopMarker_Diamond_Start                        =  20
    LaneStopMarker_Diamond_end                          =  21
    LightPost                                           =  22
    Dashed_LaneMarker_MidStart                          =  23
    Dashed_LaneMarker_MidEnd                            =  24


class TrafficSignalState(Enum):

    NO_DETECTION                 = 0
    RED                          = 1
    YELLOW                       = 2
    RED_YELLOW                   = 3
    GREEN                        = 4
    GREEN_YELLOW                 = 6
    FLASHING_BIT                 = 8  # (flashing modifier, not valid by itself)
    FLASHING_RED                 = 9
    FLASHING_YELLOW              = 10
    FLASHING_GREEN               = 12
    UP_DOWN_ARROW_BIT            = 16 # (up/down arrow modifier, not valid by itself)
    STRAIGHT_ARROW_RED           = 17
    STRAIGHT_ARROW_YELLOW        = 18
    STRAIGHT_ARROW_GREEN         = 20
    LEFT_ARROW_BIT               = 32 # (left arrow modifier, not valid by itself)
    LEFT_ARROW_RED               = 33
    LEFT_ARROW_YELLOW            = 34
    LEFT_ARROW_GREEN             = 36
    FLASHING_LEFT_ARROW_RED      = 41
    FLASHING_LEFT_ARROW_YELLOW   = 42
    FLASHING_LEFT_ARROW_GREEN    = 44
    RIGHT_ARROW_BIT              = 64 # (right arrow modifier, not valid by itself)
    RIGHT_ARROW_RED              = 65
    RIGHT_ARROW_YELLOW           = 66
    RIGHT_ARROW_GREEN            = 68
    FLASHING_RIGHT_ARROW_RED     = 73
    FLASHING_RIGHT_ARROW_YELLOW  = 74
    FLASHING_RIGHT_ARROW_GREEN   = 76
    RAILROAD_CROSSING_UNKNOWN    = 96
    RAILROAD_CROSSING_RED        = 97
    RAILROAD_CROSSING_CLEAR      = 100
    RAILROAD_CROSSING_BLOCKED    = 104
    PEDISTRIAN_SAME_DIR_UNKNOWN   = 108
    PEDISTRIAN_SAME_DIR_STOPPED   = 109
    PEDISTRIAN_SAME_DIR_WARNING   = 110
    PEDISTRIAN_SAME_DIR_CROSS     = 112
    PEDISTRIAN_PERP_DIR_UNKNOWN   = 116
    PEDISTRIAN_PERP_DIR_STOPPED   = 117
    PEDISTRIAN_PERP_DIR_WARNING   = 118
    PEDISTRIAN_PERP_DIR_CROSS     = 120
    UNKNOWN                       = 121
    OFF                           = 122


class AdjacentLane(PUBSUB_MSG_IMPL):
    e_Cnt_lane_segment_id = int
    e_e_moving_direction = MovingDirection
    e_e_lane_type = MapLaneType

    def __init__(self, e_Cnt_lane_segment_id, e_e_moving_direction, e_e_lane_type):
        # type: (int, MovingDirection, MapLaneType) -> None
        self.e_Cnt_lane_segment_id = e_Cnt_lane_segment_id
        self.e_e_moving_direction = e_e_moving_direction
        self.e_e_lane_type = e_e_lane_type

    def serialize(self):
        # type: () -> TsSYSAdjacentLane
        pubsub_msg = TsSYSAdjacentLane()

        pubsub_msg.e_Cnt_lane_segment_id = self.e_Cnt_lane_segment_id
        pubsub_msg.e_e_moving_direction = self.e_e_moving_direction
        pubsub_msg.e_e_lane_type = self.e_e_lane_type

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg):
        # type: (TsSYSAdjacentLane)->AdjacentLane
        return cls(pubsubMsg.e_Cnt_lane_segment_id, pubsubMsg.e_e_moving_direction, pubsubMsg.e_e_lane_type)


class LaneManeuver(PUBSUB_MSG_IMPL):
    e_Cnt_lane_segment_id = int
    e_e_maneuver_type = ManeuverType

    def __init__(self, e_Cnt_lane_segment_id, e_e_maneuver_type):
        # type: (int, ManeuverType) -> None
        self.e_Cnt_lane_segment_id = e_Cnt_lane_segment_id
        self.e_e_maneuver_type = e_e_maneuver_type

    def serialize(self):
        # type: () -> TsSYSLaneManeuver
        pubsub_msg = TsSYSLaneManeuver()

        pubsub_msg.e_Cnt_lane_segment_id = self.e_Cnt_lane_segment_id
        pubsub_msg.e_e_maneuver_type = self.e_e_maneuver_type

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg):
        # type: (TsSYSLaneManeuver)->LaneManeuver
        return cls(pubsubMsg.e_Cnt_lane_segment_id, pubsubMsg.e_e_maneuver_type)


class BoundaryPoint(PUBSUB_MSG_IMPL):
    e_e_lane_marker_type = MapLaneMarkerType
    e_l_s_start = float
    e_l_s_end = float

    def __init__(self, e_e_lane_marker_type, e_l_s_start, e_l_s_end):
        # type: (MapLaneMarkerType, float, float) -> None
        self.e_e_lane_marker_type = e_e_lane_marker_type
        self.e_l_s_start = e_l_s_start
        self.e_l_s_end = e_l_s_end

    def serialize(self):
        # type: () -> TsSYSBoundaryPoint
        pubsub_msg = TsSYSBoundaryPoint()

        pubsub_msg.e_e_lane_marker_type = self.e_e_lane_marker_type
        pubsub_msg.e_l_s_start = self.e_l_s_start
        pubsub_msg.e_l_s_end = self.e_l_s_end

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg):
        # type: (TsSYSBoundaryPoint)->BoundaryPoint
        return cls(pubsubMsg.e_e_lane_marker_type, pubsubMsg.e_l_s_start, pubsubMsg.e_l_s_end)


class LaneCoupling(PUBSUB_MSG_IMPL):

    e_i_lane_segment_id = int
    e_i_road_intersection_id = int
    e_i_downstream_lane_segment_id = int
    e_i_upstream_lane_segment_id = int
    e_e_maneuver_type = ManeuverType

    def __init__(self, e_i_lane_segment_id, e_i_road_intersection_id, e_i_downstream_lane_segment_id,
                 e_i_upstream_lane_segment_id, e_e_maneuver_type):
        # type: (int, int, int, int, ManeuverType) -> None
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

    def serialize(self):
        # type: () -> TsSYSLaneCoupling
        pubsub_msg = TsSYSLaneCoupling()

        pubsub_msg.e_i_lane_segment_id = self.e_i_lane_segment_id
        pubsub_msg.e_i_road_intersection_id = self.e_i_road_intersection_id
        pubsub_msg.e_i_downstream_lane_segment_id = self.e_i_downstream_lane_segment_id
        pubsub_msg.e_i_upstream_lane_segment_id = self.e_i_upstream_lane_segment_id
        pubsub_msg.e_e_maneuver_type = self.e_e_maneuver_type

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg):
        # type: (TsSYSLaneCoupling)->LaneCoupling
        return cls(pubsubMsg.e_i_lane_segment_id, pubsubMsg.e_i_road_intersection_id,
                   pubsubMsg.e_i_downstream_lane_segment_id, pubsubMsg.e_i_upstream_lane_segment_id,
                   pubsubMsg.e_e_maneuver_type)


class NominalPathPoint(PUBSUB_MSG_IMPL):
    e_l_EastX = float
    e_l_NorthY = float
    e_phi_heading = float
    e_il_curvature = float
    e_il2_curvature_rate = float
    e_phi_cross_slope = float
    e_phi_along_slope = float
    e_l_s = float
    e_l_left_offset = float
    e_l_right_offset = float

    def __init__(self, e_l_EastX, e_l_NorthY, e_phi_heading, e_il_curvature, e_il2_curvature_rate, e_phi_cross_slope,
                 e_phi_along_slope, e_l_s, e_l_left_offset, e_l_right_offset):
        # type: (float, float, float, float, float, float, float, float, float, float) -> None
        """
        Nominal (i.e. center of lane) path-point information
        :param e_l_EastX: East-X Position
        :param e_l_NorthY: North-Y Position
        :param e_phi_heading: eading/yaw of this nominal path-point, i.e. the downstream direction.
        :param e_il_curvature: Not relevant for M0
        :param e_il2_curvature_rate: Not relevant for M0
        :param e_phi_cross_slope: Not relevant for M0
        :param e_phi_along_slope: Not relevant for M0
        :param e_l_s: s-position of this nominal path-point
        :param e_l_left_offset: d-position of the left-boundary (i.e. lane edge) from this nominal path-point.
        :param e_l_right_offset: d-position of the right-boundary (i.e. lane edge) from this nominal path-point.
        """
        self.e_l_EastX = e_l_EastX
        self.e_l_NorthY = e_l_NorthY
        self.e_phi_heading = e_phi_heading
        self.e_il_curvature = e_il_curvature
        self.e_il2_curvature_rate = e_il2_curvature_rate
        self.e_phi_cross_slope = e_phi_cross_slope
        self.e_phi_along_slope = e_phi_along_slope
        self.e_l_s = e_l_s
        self.e_l_left_offset = e_l_left_offset
        self.e_l_right_offset = e_l_right_offset

    def serialize(self):
        # type: () -> TsSYSNominalPathPoint
        pubsub_msg = TsSYSNominalPathPoint()

        pubsub_msg.e_l_EastX = self.e_l_EastX
        pubsub_msg.e_l_NorthY = self.e_l_NorthY
        pubsub_msg.e_phi_heading = self.e_phi_heading
        pubsub_msg.e_il_curvature = self.e_il_curvature
        pubsub_msg.e_il2_curvature_rate = self.e_il2_curvature_rate
        pubsub_msg.e_phi_cross_slope = self.e_phi_cross_slope
        pubsub_msg.e_phi_along_slope = self.e_phi_along_slope
        pubsub_msg.e_l_s = self.e_l_s
        pubsub_msg.e_l_left_offset = self.e_l_left_offset
        pubsub_msg.e_l_right_offset = self.e_l_right_offset

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg):
        # type: (TsSYSNominalPathPoint)->NominalPathPoint
        return cls(pubsubMsg.e_l_EastX, pubsubMsg.e_l_NorthY, pubsubMsg.e_phi_heading, pubsubMsg.e_il_curvature,
                   pubsubMsg.e_il2_curvature_rate, pubsubMsg.e_phi_cross_slope, pubsubMsg.e_phi_along_slope,
                   pubsubMsg.e_l_s, pubsubMsg.e_l_left_offset, pubsubMsg.e_l_right_offset)


class StaticTrafficFlowControl(PUBSUB_MSG_IMPL):
    e_e_road_object_type = RoadObjectType
    e_l_station = float
    e_Pct_confidence = float

    def __init__(self, e_e_road_object_type, e_l_station, e_Pct_confidence):
        # type: (RoadObjectType, float, float) -> None
        """
        Static traffic-flow-control device, eg. Stop Signs (not relevant for M0)
        :param e_e_road_object_type:
        :param e_l_station:
        :param e_Pct_confidence:
        """
        self.e_e_road_object_type = e_e_road_object_type
        self.e_l_station = e_l_station
        self.e_Pct_confidence = e_Pct_confidence

    def serialize(self):
        # type: () -> TsSYSStaticTrafficFlowControl
        pubsub_msg = TsSYSStaticTrafficFlowControl()

        pubsub_msg.e_e_road_object_type = self.e_e_road_object_type
        pubsub_msg.e_l_station = self.e_l_station
        pubsub_msg.e_Pct_confidence = self.e_Pct_confidence

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg):
        # type: (TsSYSStaticTrafficFlowControl)->StaticTrafficFlowControl
        return cls(pubsubMsg.e_e_road_object_type, pubsubMsg.e_l_station,
                   pubsubMsg.e_Pct_confidence)


class DynamicStatus(PUBSUB_MSG_IMPL):
    e_e_status = TrafficSignalState
    e_Pct_confidence = float

    def __init__(self, e_e_status, e_Pct_confidence):
        # type:(TrafficSignalState, float) -> None
        """
        Status of Dynamic traffic-flow-control device, eg. red-yellow-green (not relevant for M0)
        :param e_e_status:
        :param e_Pct_confidence:
        """
        self.e_e_status = e_e_status
        self.e_Pct_confidence = e_Pct_confidence

    def serialize(self):
        # type: () -> TsSYSDynamicStatus
        pubsub_msg = TsSYSDynamicStatus()

        pubsub_msg.e_e_status = self.e_e_status
        pubsub_msg.e_Pct_confidence = self.e_Pct_confidence

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg):
        # type: (TsSYSDynamicStatus)->DynamicStatus
        return cls(pubsubMsg.e_e_status, pubsubMsg.e_Pct_confidence)


class DynamicTrafficFlowControl(PUBSUB_MSG_IMPL):
    e_e_road_object_type = RoadObjectType
    e_l_station = float
    e_Cnt_dynamic_status_count = int
    as_dynamic_status = List[DynamicStatus]

    def __init__(self, e_e_road_object_type, e_l_station, e_Cnt_dynamic_status_count, as_dynamic_status):
        # type: (RoadObjectType, float, int, List[DynamicStatus]) -> None
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

    def serialize(self):
        # type: () -> TsSYSDynamicTrafficFlowControl
        pubsub_msg = TsSYSDynamicTrafficFlowControl()

        pubsub_msg.e_e_road_object_type = self.e_e_road_object_type
        pubsub_msg.e_l_station = self.e_l_station
        pubsub_msg.e_Cnt_dynamic_status_count = self.e_Cnt_dynamic_status_count

        pubsub_msg.as_dynamic_status = list()
        for i in range(pubsub_msg.e_Cnt_dynamic_status_count):
            pubsub_msg.as_dynamic_status.append(self.as_dynamic_status[i].serialize())

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg):
        # type: (TsSYSDynamicTrafficFlowControl) -> DynamicTrafficFlowControl
        dynamic_statuses = list()
        for i in range(pubsubMsg.e_Cnt_dynamic_status_count):
            dynamic_statuses.append(DynamicTrafficFlowControl.deserialize(pubsubMsg.as_dynamic_status[i]))
        return cls(pubsubMsg.e_e_road_object_type, pubsubMsg.e_l_station, pubsubMsg.e_Cnt_dynamic_status_count,
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
    as_downstream_lanes = List[LaneManeuver]
    e_Cnt_upstream_lane_count = int
    as_upstream_lanes = List[LaneManeuver]
    e_v_nominal_speed = float
    e_Cnt_nominal_path_point_count = int
    a_nominal_path_points = List[NominalPathPoint]
    e_Cnt_left_boundary_points_count = int
    as_left_boundary_points = List[BoundaryPoint]
    e_Cnt_right_boundary_points_count = int
    as_right_boundary_points = List[BoundaryPoint]
    e_i_downstream_road_intersection_id = int
    e_Cnt_lane_coupling_count = int
    as_lane_coupling = List[LaneCoupling]

    def __init__(self, e_i_lane_segment_id, e_i_road_segment_id, e_e_lane_type, e_Cnt_static_traffic_flow_control_count,
                 as_static_traffic_flow_control, e_Cnt_dynamic_traffic_flow_control_count, as_dynamic_traffic_flow_control,
                 e_Cnt_left_adjacent_lane_count, as_left_adjacent_lanes, e_Cnt_right_adjacent_lane_count,
                 as_right_adjacent_lanes, e_Cnt_downstream_lane_count, as_downstream_lanes, e_Cnt_upstream_lane_count,
                 as_upstream_lanes, e_v_nominal_speed, e_Cnt_nominal_path_point_count, a_nominal_path_points,
                 e_Cnt_left_boundary_points_count, as_left_boundary_points, e_Cnt_right_boundary_points_count,
                 as_right_boundary_points, e_i_downstream_road_intersection_id, e_Cnt_lane_coupling_count,
                 as_lane_coupling):
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

    def serialize(self):
        # type: () -> TsSYSSceneLaneSegment
        pubsub_msg = TsSYSSceneLaneSegment()

        pubsub_msg.e_i_lane_segment_id = self.e_i_lane_segment_id
        pubsub_msg.e_i_road_segment_id = self.e_i_road_segment_id
        pubsub_msg.e_e_lane_type = self.e_e_lane_type
        pubsub_msg.e_Cnt_static_traffic_flow_control_count = self.e_Cnt_static_traffic_flow_control_count
        pubsub_msg.as_static_traffic_flow_control = list()
        for i in range(pubsub_msg.e_Cnt_static_traffic_flow_control_count):
            pubsub_msg.as_static_traffic_flow_control.append(self.as_static_traffic_flow_control[i].serialize())

        pubsub_msg.e_Cnt_dynamic_traffic_flow_control_count = self.e_Cnt_dynamic_traffic_flow_control_count
        pubsub_msg.as_dynamic_traffic_flow_control = list()
        for i in range(pubsub_msg.e_Cnt_dynamic_traffic_flow_control_count):
            pubsub_msg.as_dynamic_traffic_flow_control.append(self.as_dynamic_traffic_flow_control[i].serialize())

        pubsub_msg.e_Cnt_left_adjacent_lane_count = self.e_Cnt_left_adjacent_lane_count
        pubsub_msg.as_left_adjacent_lanes = list()
        for i in range(pubsub_msg.e_Cnt_left_adjacent_lane_count):
            pubsub_msg.as_left_adjacent_lanes.append(self.as_left_adjacent_lanes[i].serialize())

        pubsub_msg.e_Cnt_right_adjacent_lane_count = self.e_Cnt_right_adjacent_lane_count
        pubsub_msg.as_right_adjacent_lanes = list()
        for i in range(pubsub_msg.e_Cnt_right_adjacent_lane_count):
            pubsub_msg.as_right_adjacent_lanes.append(self.as_right_adjacent_lanes[i].serialize())

        pubsub_msg.e_Cnt_downstream_lane_count = self.e_Cnt_downstream_lane_count
        pubsub_msg.as_downstream_lanes = self.as_downstream_lanes

        pubsub_msg.e_Cnt_upstream_lane_count = self.e_Cnt_upstream_lane_count
        pubsub_msg.as_upstream_lanes = self.as_upstream_lanes

        pubsub_msg.e_v_nominal_speed = self.e_v_nominal_speed
        pubsub_msg.e_Cnt_nominal_path_point_count = self.e_Cnt_nominal_path_point_count
        pubsub_msg.a_nominal_path_points = self.a_nominal_path_points

        pubsub_msg.e_Cnt_left_boundary_points_count = self.e_Cnt_left_boundary_points_count
        pubsub_msg.as_left_boundary_points = self.as_left_boundary_points

        pubsub_msg.e_Cnt_right_boundary_points_count = self.e_Cnt_right_boundary_points_count
        pubsub_msg.as_right_boundary_points = self.as_right_boundary_points

        pubsub_msg.e_i_downstream_road_intersection_id = self.e_i_downstream_road_intersection_id
        pubsub_msg.e_Cnt_lane_coupling_count = self.e_Cnt_lane_coupling_count
        pubsub_msg.as_lane_coupling = self.as_lane_coupling


        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg):
        # type: (TsSYSSceneLaneSegment) -> SceneLaneSegment
        dynamic_statuses = list()
        for i in range(pubsubMsg.e_Cnt_dynamic_status_count):
            dynamic_statuses.append(DynamicTrafficFlowControl.deserialize(pubsubMsg.as_dynamic_status[i]))
        return cls(pubsubMsg.e_e_road_object_type, pubsubMsg.e_l_station, pubsubMsg.e_Cnt_dynamic_status_count,
                   dynamic_statuses)

class DataSceneStatic(PUBSUB_MSG_IMPL):
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
        # type: () -> TsSYSDataSceneStatic
        pubsub_msg = TsSYSDataSceneStatic()

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
        # type: (TsSYSDataSceneStatic) -> DataSceneStatic

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


class SceneStatic(PUBSUB_MSG_IMPL):
    s_Header = Header
    s_MapOrigin = MapOrigin
    s_Data = DataSceneStatic

    def __init__(self, s_Header, s_MapOrigin, s_Data):
        # type: (Header, MapOrigin, DataSceneStatic) -> None
        self.s_Header = s_Header
        self.s_MapOrigin = s_MapOrigin
        self.s_Data = s_Data

    def serialize(self):
        # type: () -> TsSYSSceneStatic
        pubsub_msg = TsSYSSceneStatic()
        pubsub_msg.s_Header = self.s_Header.serialize()
        pubsub_msg.s_MapOrigin = self.s_MapOrigin.serialize()
        pubsub_msg.s_Data = self.s_Data.serialize()
        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg):
        # type: (TsSYSSceneStatic) -> SceneStatic
        return cls(Header.deserialize(pubsubMsg.s_Header),
                   MapOrigin.deserialize(pubsubMsg.s_MapOrigin),
                   DataSceneStatic.deserialize(pubsubMsg.s_Data))

