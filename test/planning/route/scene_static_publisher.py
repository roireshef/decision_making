from time import time
from typing import List
import numpy as np

from common_data.interface.Rte_Types.python import Rte_Types_pubsub as pubsub_topics

from decision_making.src.messages.scene_common_messages import Header, Timestamp, MapOrigin
from decision_making.src.messages.scene_static_message import (
    SceneStatic,
    DataSceneStatic,
    SceneStaticGeometry,
    SceneStaticBase,
    NavigationPlan,
    SceneLaneSegmentGeometry,
    BoundaryPoint,
    SceneLaneSegmentBase,
    SceneRoadIntersection,
    SceneRoadSegment,
    MAX_NOMINAL_PATH_POINT_FIELDS,
    StaticTrafficFlowControl,
    DynamicTrafficFlowControl,
    DynamicStatus,
    AdjacentLane,
    LaneSegmentConnectivity,
    LaneCoupling)
from decision_making.src.messages.scene_static_enums import (
    MapLaneMarkerType,
    MapRoadSegmentType,
    MapLaneType,
    RoadObjectType,
    TrafficSignalState,
    MovingDirection,
    ManeuverType,
    LaneMappingStatusType,
    MapLaneDirection,
    GMAuthorityType,
    LaneConstructionType)

class SceneStaticPublisher:
    """ TODO: Add description """

    def generate_data(self, num_road_segments: int, road_segment_ids: List[int], num_lane_segments: int, lane_segment_ids: List[List[int]],
                      navigation_plan: List[int]) -> SceneStatic:
        """ TODO: Add description """

        # Time since the epoch
        timestamp_object = Timestamp.from_seconds(time())

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
                                   s_SceneStaticBase=SceneStaticBase(e_Cnt_num_lane_segments=num_lane_segments,
                                                                     as_scene_lane_segments=self._generate_lane_segments(road_segment_ids,
                                                                                                                         num_lane_segments,
                                                                                                                         lane_segment_ids),
                                                                     e_Cnt_num_road_intersections=0,
                                                                     as_scene_road_intersection=self._generate_road_intersections(),
                                                                     e_Cnt_num_road_segments=num_road_segments,
                                                                     as_scene_road_segment=self._generate_road_segments(road_segment_ids,
                                                                                                                        lane_segment_ids)),
                                   s_SceneStaticGeometry=SceneStaticGeometry(e_Cnt_num_lane_segments=0,
                                                                             as_scene_lane_segments=self._generate_geometry()),
                                   s_NavigationPlan=NavigationPlan(e_Cnt_num_road_segments=num_road_segments,
                                                                   a_i_road_segment_ids=np.array(navigation_plan, dtype=np.int))))

    def _generate_lane_segments(self, road_segment_ids: List[int] = None, num_lane_segments: int = 1,
                                lane_segment_ids: List[List[int]] = None) -> List[SceneLaneSegmentBase]:
        """
        Generates default lane segment geometry data

        Args:
            road_segment_ids: List of road segment IDs, Default is None
            num_lane_segments: Number of lane segments, Default is 1
            lane_segment_ids: List of lane segment IDs, Default is None
        """
        # Add error catching in case lists are None

        lane_segment_lite = []

        for i in range(len(road_segment_ids)):
            for j in range(len(lane_segment_ids[0])):
                if i == len(road_segment_ids) - 1:
                    downstream_lane_count = 0
                    downstream_lanes = self._generate_lane_segment_connectivity()
                else:
                    downstream_lane_count = 1
                    downstream_lanes = [LaneSegmentConnectivity(e_i_lane_segment_id=lane_segment_ids[i+1][j],
                                                                e_e_maneuver_type=ManeuverType.STRAIGHT_CONNECTION)]

                num_active_lane_attributes = 4
                active_lane_attribute_indices = np.array([0, 1, 2, 3], dtype=np.int)
                lane_attributes = np.array([LaneMappingStatusType.CeSYS_e_LaneMappingStatusType_HDMap.value,
                                         GMAuthorityType.CeSYS_e_GMAuthorityType_None.value,
                                         LaneConstructionType.CeSYS_e_LaneConstructionType_Normal.value,
                                         MapLaneDirection.CeSYS_e_MapLaneDirection_SameAs_HostVehicle.value], dtype=np.int)
                lane_attribute_confidences = np.ones(4, dtype=np.float)
                
                lane_segment_lite.append(SceneLaneSegmentBase(e_i_lane_segment_id=lane_segment_ids[i][j],
                                                              e_i_road_segment_id=road_segment_ids[i],
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
                                                              e_v_nominal_speed=0,
                                                              e_i_downstream_road_intersection_id=1,
                                                              e_Cnt_lane_coupling_count=0,
                                                              as_lane_coupling=self._generate_lane_coupling(),
                                                              e_l_length=100,
                                                              e_Cnt_num_active_lane_attributes=num_active_lane_attributes,
                                                              a_i_active_lane_attribute_indices=active_lane_attribute_indices,
                                                              a_cmp_lane_attributes=lane_attributes,
                                                              a_cmp_lane_attribute_confidences=lane_attribute_confidences))
        
        return lane_segment_lite

    def _generate_road_segments(self, road_segment_ids: List[int] = None, lane_segment_ids: List[List[int]] = None) -> \
        List[SceneRoadSegment]:
        """
        Generates default road segment data

        Args:
            road_segment_ids: List of road segment IDs, Default is None
            lane_segment_ids: List of lane segment IDs, Default is None
        """
        # Add error catching in case lists are None

        return [SceneRoadSegment(e_i_road_segment_id=road_segment_ids[i],
                                 e_i_road_id=1,
                                 e_Cnt_lane_segment_id_count=len(lane_segment_ids),
                                 a_i_lane_segment_ids=np.array(lane_segment_ids[i], dtype=np.int),
                                 e_e_road_segment_type=MapRoadSegmentType.Normal,
                                 e_Cnt_upstream_segment_count=0,
                                 a_i_upstream_road_segment_ids=np.array([0], dtype=np.int),
                                 e_Cnt_downstream_segment_count=0,
                                 a_i_downstream_road_segment_ids=np.array([0], dtype=np.int)) \
                for i in range(len(road_segment_ids))]
    
    def _generate_road_intersections(self, num_intersections: int = 1) -> List[SceneRoadIntersection]:
        """
        Generates default road intersection data

        Args:
            num_intersections: Number of intersections, Default is 1
        """
        return [SceneRoadIntersection(e_i_road_intersection_id=1,
                                      e_Cnt_lane_coupling_count=0,
                                      a_i_lane_coupling_segment_ids=0,
                                      e_Cnt_intersection_road_segment_count=0,
                                      a_i_intersection_road_segment_ids=0)]
    
    def _generate_lane_coupling(self) -> List[LaneCoupling]:
        """
        Generates default lane coupling data
        """
        return [LaneCoupling(e_i_lane_segment_id=1,
                             e_i_road_intersection_id=1,
                             e_i_downstream_lane_segment_id=1,
                             e_i_upstream_lane_segment_id=1,
                             e_e_maneuver_type=ManeuverType.STRAIGHT_CONNECTION)]
    
    def _generate_lane_segment_connectivity(self) -> List[LaneSegmentConnectivity]:
        """
        Generates default lane segment connectivity data
        """
        return [LaneSegmentConnectivity(e_i_lane_segment_id=1,
                                        e_e_maneuver_type=ManeuverType.STRAIGHT_CONNECTION)]
    
    def _generate_adjacent_lane(self) -> List[AdjacentLane]:
        """
        Generates default adjacent lane data
        """
        return [AdjacentLane(e_i_lane_segment_id=1,
                             e_e_moving_direction=0,
                             e_e_lane_type=MovingDirection.Adjacent_or_same_dir)]
    
    def _generate_dynamic_status(self) -> List[DynamicStatus]:
        """
        Generates default dynamic status data
        """
        return [DynamicStatus(e_e_status=TrafficSignalState.NO_DETECTION,
                              e_Pct_confidence=0)]
    
    def _generate_dynamic_traffic_flow_control(self) -> List[DynamicTrafficFlowControl]:
        """
        Generates default dynamic traffic flow control data
        """
        return [DynamicTrafficFlowControl(e_e_road_object_type=RoadObjectType.Yield,
                                          e_l_station=0,
                                          e_Cnt_dynamic_status_count=0,
                                          as_dynamic_status=self._generate_dynamic_status())]
    
    def _generate_traffic_flow_control(self) -> List[StaticTrafficFlowControl]:
        """
        Generates default traffic flow control data
        """
        return [StaticTrafficFlowControl(e_e_road_object_type=RoadObjectType.Yield,
                                         e_l_station=0,
                                         e_Pct_confidence=0)]
    
    def _generate_geometry(self, num_nominal_path_points: int = 1) -> List[SceneLaneSegmentGeometry]:
        """
        Generates default lane segment geometry data

        Args:
            num_nominal_path_points: Number of nominal path points, Default is 1
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
