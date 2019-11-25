from time import time
from typing import List, Dict, Tuple
import numpy as np
from decision_making.src.messages.scene_common_messages import Header, Timestamp, MapOrigin
from decision_making.src.messages.scene_static_enums import MapLaneMarkerType, MapRoadSegmentType,\
    MapLaneType, RoadObjectType, TrafficSignalState, MovingDirection, ManeuverType, LaneMappingStatusType,\
    MapLaneDirection, GMAuthorityType, LaneConstructionType, RoutePlanLaneSegmentAttr
from decision_making.src.messages.scene_static_message import SceneStatic, DataSceneStatic,\
    SceneStaticGeometry, SceneStaticBase, NavigationPlan, SceneLaneSegmentGeometry,\
    BoundaryPoint, SceneLaneSegmentBase, SceneRoadSegment,\
    MAX_NOMINAL_PATH_POINT_FIELDS, StaticTrafficFlowControl, DynamicTrafficFlowControl,\
    DynamicStatus, AdjacentLane, LaneSegmentConnectivity, LaneOverlap, LaneOverlapType


class SceneStaticPublisher:
    RoadSegmentID = int
    LaneSegmentID = int
    DownstreamRoadSegmentIDs = Dict[RoadSegmentID, List[RoadSegmentID]]
    LaneConnectivity = Tuple[LaneSegmentID, ManeuverType]
    DownstreamLaneConnectivity = Dict[LaneSegmentID, List[LaneConnectivity]]
    IsLaneAttributeActive = bool
    LaneAttribute = int # actually, LaneMappingStatusType, MapLaneDirection, GMAuthorityType, or LaneConstructionType
    LaneAttributeConfidence = float
    LaneAttributeModification = Tuple[IsLaneAttributeActive, RoutePlanLaneSegmentAttr, LaneAttribute, LaneAttributeConfidence]
    LaneAttributeModifications = Dict[LaneSegmentID, List[LaneAttributeModification]]

    def __init__(self, road_segment_ids: List[RoadSegmentID], lane_segment_ids: List[List[LaneSegmentID]],
                 navigation_plan: List[RoadSegmentID], downstream_road_segment_ids: DownstreamRoadSegmentIDs = None,
                 downstream_lane_connectivity: DownstreamLaneConnectivity = None,
                 lane_attribute_modifications: LaneAttributeModifications = None):
        """
        Creates SCENE_STATIC message for testing
        :param road_segment_ids: Array of road segment IDs
        :param lane_segment_ids: 2D array containing lane segment IDs. Each row is associated with a road segment
        :param navigation_plan: Ordered sequence of road segment IDs that makes up the navigaiton plan
        :param downstream_road_segment_ids: Optional dictionary containing downstream road segment IDs
        :param downstream_lane_connectivity: Optional dictionary containing downstream lane connectivity information
        :param lane_attribute_modifications: Optional dictionary containing modifications to the default lane attributes
        """
        if downstream_road_segment_ids is None:
            downstream_road_segment_ids = {}

        if downstream_lane_connectivity is None:
            downstream_lane_connectivity = {}

        if lane_attribute_modifications is None:
            lane_attribute_modifications = {}

        self._road_segment_ids = road_segment_ids
        self._lane_segment_ids = lane_segment_ids
        self._navigation_plan = navigation_plan
        self._downstream_road_segment_ids = downstream_road_segment_ids
        self._downstream_lane_connectivity = downstream_lane_connectivity
        self._lane_attribute_modifications = lane_attribute_modifications

    def generate_data(self) -> SceneStatic:
        """ Generates scene static data """
        # Time since the epoch
        timestamp_object = Timestamp.from_seconds(time())

        lane_geometry = self._generate_geometry()
        a_nominal_path_points = np.concatenate([l.a_nominal_path_points for l in lane_geometry], axis=0)
        return SceneStatic(
            s_Header=Header(e_Cnt_SeqNum=0,
                            s_Timestamp=timestamp_object,
                            e_Cnt_version=0),
            s_Data=DataSceneStatic(e_b_Valid=True,
                                   s_RecvTimestamp=timestamp_object,
                                   e_l_perception_horizon_front=50,
                                   e_l_perception_horizon_rear=50,
                                   s_MapOrigin=MapOrigin(e_phi_latitude=0,
                                                         e_phi_longitude=0,
                                                         e_l_altitude=0,
                                                         s_Timestamp=timestamp_object),
                                   s_SceneStaticBase=SceneStaticBase(e_Cnt_num_lane_segments=
                                                                        sum(len(row) for row in self._lane_segment_ids),
                                                                     as_scene_lane_segments=self._generate_lane_segments(),
                                                                     e_Cnt_num_road_segments=len(self._road_segment_ids),
                                                                     as_scene_road_segment=self._generate_road_segments()),
                                   s_SceneStaticGeometry=SceneStaticGeometry(e_Cnt_num_lane_segments=0,
                                                                             as_scene_lane_segments=lane_geometry,
                                                                             a_nominal_path_points=a_nominal_path_points),
                                   s_NavigationPlan=NavigationPlan(e_Cnt_num_road_segments=len(self._navigation_plan),
                                                                   a_i_road_segment_ids=np.array(self._navigation_plan))))

    def _generate_lane_segments(self) -> List[SceneLaneSegmentBase]:
        """ Generates default lane segment geometry data """
        lane_segment_base = []

        for i, road_segment_id in enumerate(self._road_segment_ids):
            for j, lane_segment_id in enumerate(self._lane_segment_ids[i]):
                if bool(self._downstream_lane_connectivity):
                    # _downstream_lane_connectivity is NOT empty
                    if lane_segment_id in self._downstream_lane_connectivity:
                        downstream_lane_count = len(self._downstream_lane_connectivity[lane_segment_id])
                        downstream_lanes = [LaneSegmentConnectivity(e_i_lane_segment_id=downstream_lane[0],
                                                                    e_e_maneuver_type=downstream_lane[1])
                                            for downstream_lane in self._downstream_lane_connectivity[lane_segment_id]]
                    else:
                        downstream_lane_count = 0
                        downstream_lanes = self._generate_lane_segment_connectivity()
                else:
                    # _downstream_lane_connectivity is empty
                    if road_segment_id is self._road_segment_ids[-1]:
                        downstream_lane_count = 0
                        downstream_lanes = self._generate_lane_segment_connectivity()
                    else:
                        downstream_lane_count = 1
                        downstream_lanes = [LaneSegmentConnectivity(e_i_lane_segment_id=self._lane_segment_ids[i+1][j],
                                                                    e_e_maneuver_type=ManeuverType.STRAIGHT_CONNECTION)]

                # Default lane attribute values
                num_active_lane_attributes = 4
                active_lane_attribute_indices = np.array([0, 1, 2, 3])
                lane_attributes = [LaneMappingStatusType.CeSYS_e_LaneMappingStatusType_HDMap,
                                   GMAuthorityType.CeSYS_e_GMAuthorityType_None,
                                   LaneConstructionType.CeSYS_e_LaneConstructionType_Normal,
                                   MapLaneDirection.CeSYS_e_MapLaneDirection_SameAs_HostVehicle]
                lane_attribute_confidences = np.ones(4)

                # Check for lane attribute modifications
                if lane_segment_id in self._lane_attribute_modifications:
                    for lane_modification in self._lane_attribute_modifications[lane_segment_id]:
                        if lane_modification[0] is True:
                            lane_attributes[lane_modification[1]] = lane_modification[2]
                            lane_attribute_confidences[lane_modification[1]] = lane_modification[3]
                        else:
                            active_lane_attribute_indices = np.delete(active_lane_attribute_indices, lane_modification[1])
                            num_active_lane_attributes -= 1

                lane_segment_base.append(SceneLaneSegmentBase(e_i_lane_segment_id=lane_segment_id,
                                                              e_i_road_segment_id=road_segment_id,
                                                              e_e_lane_type=MapLaneType.ControlledAccess_DividedRoadLane,
                                                              e_Cnt_static_traffic_flow_control_count=0,
                                                              as_static_traffic_flow_control=self._generate_traffic_flow_control(),
                                                              e_Cnt_dynamic_traffic_flow_control_count=0,
                                                              as_dynamic_traffic_flow_control=self._generate_dynamic_traffic_flow_control(),
                                                              e_Cnt_left_adjacent_lane_count=0,
                                                              as_left_adjacent_lanes=self._generate_adjacent_lane(),
                                                              e_Cnt_right_adjacent_lane_count=0,
                                                              as_right_adjacent_lanes=self._generate_adjacent_lane(),
                                                              e_Cnt_downstream_lane_count=downstream_lane_count,
                                                              as_downstream_lanes=downstream_lanes,
                                                              e_Cnt_upstream_lane_count=0,
                                                              as_upstream_lanes=self._generate_lane_segment_connectivity(),
                                                              e_v_nominal_speed=20,
                                                              e_l_length=100,
                                                              e_Cnt_num_active_lane_attributes=num_active_lane_attributes,
                                                              a_i_active_lane_attribute_indices=active_lane_attribute_indices,
                                                              a_cmp_lane_attributes=lane_attributes,
                                                              a_cmp_lane_attribute_confidences=lane_attribute_confidences,
                                                              e_Cnt_lane_overlap_count=0,
                                                              as_lane_overlaps=self._generate_lane_overlap()))

        return lane_segment_base

    def _generate_road_segments(self) -> List[SceneRoadSegment]:
        """ Generates road segment data """
        if bool(self._downstream_road_segment_ids):
            # _downstream_road_segment_ids is NOT empty
            return [SceneRoadSegment(e_i_road_segment_id=road_segment_id,
                                     e_Cnt_lane_segment_id_count=len(self._lane_segment_ids[i]),
                                     a_i_lane_segment_ids=np.array(self._lane_segment_ids[i]),
                                     e_e_road_segment_type=MapRoadSegmentType.Normal,
                                     e_Cnt_upstream_segment_count=0,
                                     a_i_upstream_road_segment_ids=np.array([0]),
                                     e_Cnt_downstream_segment_count=len(self._downstream_road_segment_ids[road_segment_id]),
                                     a_i_downstream_road_segment_ids=np.array(self._downstream_road_segment_ids[road_segment_id])) \
                    for i, road_segment_id in enumerate(self._road_segment_ids)]
        else:
            # _downstream_road_segment_ids is empty
            return [SceneRoadSegment(e_i_road_segment_id=road_segment_id,
                                     e_Cnt_lane_segment_id_count=len(self._lane_segment_ids[i]),
                                     a_i_lane_segment_ids=np.array(self._lane_segment_ids[i]),
                                     e_e_road_segment_type=MapRoadSegmentType.Normal,
                                     e_Cnt_upstream_segment_count=0,
                                     a_i_upstream_road_segment_ids=np.array([0]),
                                     e_Cnt_downstream_segment_count=0,
                                     a_i_downstream_road_segment_ids=np.array([]))
                    for i, road_segment_id in enumerate(self._road_segment_ids)]

    def _generate_lane_segment_connectivity(self) -> List[LaneSegmentConnectivity]:
        """ Generates default lane segment connectivity data """
        return [LaneSegmentConnectivity(e_i_lane_segment_id=1,
                                        e_e_maneuver_type=ManeuverType.STRAIGHT_CONNECTION)]

    def _generate_adjacent_lane(self) -> List[AdjacentLane]:
        """ Generates default adjacent lane data """
        return [AdjacentLane(e_i_lane_segment_id=1,
                             e_e_moving_direction=0,
                             e_e_lane_type=MovingDirection.Adjacent_or_same_dir)]

    def _generate_dynamic_status(self) -> List[DynamicStatus]:
        """ Generates default dynamic status data """
        return [DynamicStatus(e_e_status=TrafficSignalState.NO_DETECTION,
                              e_Pct_confidence=0)]

    def _generate_dynamic_traffic_flow_control(self) -> List[DynamicTrafficFlowControl]:
        """ Generates default dynamic traffic flow control data """
        return [DynamicTrafficFlowControl(e_e_road_object_type=RoadObjectType.Yield,
                                          e_l_station=0,
                                          e_Cnt_dynamic_status_count=0,
                                          as_dynamic_status=self._generate_dynamic_status())]

    def _generate_traffic_flow_control(self) -> List[StaticTrafficFlowControl]:
        """ Generates default traffic flow control data """
        return [StaticTrafficFlowControl(e_e_road_object_type=RoadObjectType.Yield,
                                         e_l_station=0,
                                         e_Pct_confidence=0)]

    def _generate_lane_overlap(self) -> List[LaneOverlap]:
        """ Generate default lane overlap data"""
        return [LaneOverlap(e_i_other_lane_segment_id=0,
                            a_l_source_lane_overlap_stations=np.array([0,0]),
                            a_l_other_lane_overlap_stations=np.array([0,0]),
                            e_e_lane_overlap_type=LaneOverlapType.CeSYS_e_LaneOverlapType_Unknown)]


    def _generate_geometry(self, num_nominal_path_points: int = 1) -> List[SceneLaneSegmentGeometry]:
        """
        Generates default lane segment geometry data
        :param num_nominal_path_points: Number of nominal path points, Default is 1
        """
        boundary_point = [BoundaryPoint(e_e_lane_marker_type=MapLaneMarkerType.MapLaneMarkerType_None,
                                        e_l_s_start=0,
                                        e_l_s_end=0)]

        return [SceneLaneSegmentGeometry(e_i_lane_segment_id=0,
                                         e_i_road_segment_id=0,
                                         e_Cnt_nominal_path_point_count=num_nominal_path_points,
                                         a_nominal_path_points=np.zeros((num_nominal_path_points, MAX_NOMINAL_PATH_POINT_FIELDS)),
                                         e_Cnt_left_boundary_points_count=1,
                                         as_left_boundary_points=boundary_point,
                                         e_Cnt_right_boundary_points_count=1,
                                         as_right_boundary_points=boundary_point)]
