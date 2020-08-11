from enum import IntEnum, Enum


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


class LaneOverlapType(Enum):
    CeSYS_e_LaneOverlapType_Unknown = 0
    CeSYS_e_LaneOverlapType_Cross = 1
    CeSYS_e_LaneOverlapType_Merge = 2
    CeSYS_e_LaneOverlapType_Split = 3


class MapRoadSegmentType(Enum):
    Normal = 0
    Intersection = 1
    TurnOnly = 2
    Unknown = 3


class MovingDirection(Enum):
    Invalid = 0            # Used until mapping fix their output to have the correct MovingDirection
    Adjacent_or_same_dir = 1
    Opposing = 2
    Centerlane = 3


class ManeuverType(Enum):
    UNKNOWN = 0
    STRAIGHT_CONNECTION = 1
    LEFT_TURN_CONNECTION = 2
    RIGHT_TURN_CONNECTION = 3
    LEFT_MERGE_CONNECTION = 4
    RIGHT_MERGE_CONNECTION = 5
    LEFT_EXIT_CONNECTION = 6
    RIGHT_EXIT_CONNECTION = 7
    LEFT_FORK_CONNECTION = 8
    RIGHT_FORK_CONNECTION = 9
    TRAFFIC_CIRCLE_CONNECTION = 10
    LEFT_LANE_CHANGE_CONNECTION = 11
    RIGHT_LANE_CHANGE_CONNECTION = 12
    NO_ALLOWED_CONNECTION = 13
    LEFT_SPLIT = 14
    RIGHT_SPLIT = 15


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
    Bike_Pedestrian = 25


class StaticTrafficControlDeviceType(Enum):
    UNKNOWN = 0          # Type of control not mapped
    YIELD = 1            # Host must yield to conflicting traffic
    STOP = 2             # Host must stop, then yield to conflicting traffic (e.g. side-road to main road intersection)
    CROSSWALK = 3        # Point we should yield to vulnerable road users
    MOVABLE_BARRIER = 4  # A gate or barrier may prevent passage at certain times?


class DynamicTrafficControlDeviceType(Enum):
    UNKNOWN = 0             # Type of control not mapped
    TRAFFIC_LIGHT = 1       # Traffic light with no direction indicator for protected phase
    RAILROAD_CROSSING = 2   # Slower speed if associated TCD light is active/flashing?
    SCHOOL_ZONE = 3         # Slower speed if associated TCD light is active/flashing?


class TrafficLightState(Enum):
    UNKNOWN = 0
    GREEN = 1
    GREEN_OLD = 4   # In simulation we get GREEN on old value # TODO: remove this value once simulation mode is change
    NOT_GREEN = 2


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


class RoutePlanLaneSegmentAttr(IntEnum):
    CeSYS_e_RoutePlanLaneSegmentAttr_MappingStatus = 0
    CeSYS_e_RoutePlanLaneSegmentAttr_GMFA = 1
    CeSYS_e_RoutePlanLaneSegmentAttr_Construction = 2
    CeSYS_e_RoutePlanLaneSegmentAttr_Direction = 3


class LaneMappingStatusType(IntEnum):
    CeSYS_e_LaneMappingStatusType_NotMapped = 0
    CeSYS_e_LaneMappingStatusType_HDMap = 1
    CeSYS_e_LaneMappingStatusType_MDMap = 2
    CeSYS_e_LaneMappingStatusType_CameraMap = 4
    CeSYS_e_LaneMappingStatusType_RadarMap = 8
    CeSYS_e_LaneMappingStatusType_LidarMap = 16
    CeSYS_e_LaneMappingStatusType_METAMap = 32
    CeSYS_e_LaneMappingStatusType_Obsolete = 128


class MapLaneDirection(IntEnum):
    CeSYS_e_MapLaneDirection_SameAs_HostVehicle = 0
    CeSYS_e_MapLaneDirection_OppositeTo_HostVehicle = 1
    CeSYS_e_MapLaneDirection_Left_Towards_HostVehicle = 2
    CeSYS_e_MapLaneDirection_Left_AwayFrom_HostVehicle = 3
    CeSYS_e_MapLaneDirection_Right_Towards_HostVehicle = 4
    CeSYS_e_MapLaneDirection_Right_AwayFrom_HostVehicle = 5


class GMAuthorityType(IntEnum):
    CeSYS_e_GMAuthorityType_None = 0
    CeSYS_e_GMAuthorityType_RoadConstruction = 1
    CeSYS_e_GMAuthorityType_BadRoadCondition = 2
    CeSYS_e_GMAuthorityType_ComplexRoad = 3
    CeSYS_e_GMAuthorityType_MovableBarriers = 4
    CeSYS_e_GMAuthorityType_BidirectionalFreew = 5
    CeSYS_e_GMAuthorityType_HighCrossTrackSlope = 6
    CeSYS_e_GMAuthorityType_HighAlongTrackSlope = 7
    CeSYS_e_GMAuthorityType_HighVerticalCurvature = 8
    CeSYS_e_GMAuthorityType_HighHorizontalCurvat = 9
    CeSYS_e_GMAuthorityType_Unknown = 10


class LaneConstructionType(IntEnum):
    CeSYS_e_LaneConstructionType_Normal = 0
    CeSYS_e_LaneConstructionType_Blocked = 1
    CeSYS_e_LaneConstructionType_HalfBlocked = 2
    CeSYS_e_LaneConstructionType_Unknown = 3


class LineSegmentType(IntEnum):
    CeSYS_LineSegmentType_Unknown = 0
    CeSYS_LineSegmentType_LaneEdge = 1
    CeSYS_LineSegmentType_RoadEdge = 2
    CeSYS_LineSegmentType_RoadBarrier = 3
    CeSYS_LineSegmentType_NominalPath = 4


class LineSegmentAttrType(IntEnum):
    CeSYS_LineSegmentAttr_Unknown = 0
    CeSYS_LineSegmentAttr_GpsCover = 1
    CeSYS_LineSegmentAttr_PaintColor = 2
