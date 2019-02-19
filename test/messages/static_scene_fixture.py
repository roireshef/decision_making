import pytest
import numpy as np
from typing import Dict, List, Tuple

from decision_making.src.messages.scene_common_messages import Header, MapOrigin, Timestamp
from decision_making.src.messages.scene_static_message import SceneStatic, SceneStaticBase, SceneStaticGeometry, NavigationPlan, \
    SceneRoadSegment, MapRoadSegmentType, SceneLaneSegmentGeometry, SceneLaneSegmentBase, DataSceneStatic, \
    MapLaneType, LaneSegmentConnectivity, ManeuverType, NominalPathPoint, MapLaneMarkerType, BoundaryPoint, AdjacentLane, MovingDirection
from decision_making.src.planning.types import FP_SX, FP_DX
from decision_making.src.messages.scene_static_enums import (
    LaneMappingStatusType,
    GMAuthorityType,
    LaneConstructionType,
    MapLaneDirection)

from mapping.src.exceptions import NextRoadNotFound
from mapping.src.model.map_api import MapAPI
from mapping.src.service.map_service import MapService


def get_connectivity_lane_segment(map_api, road_segment_id, lane_ordinal):
    map_model = map_api._cached_map_model
    try:
        prev_road_id = map_model.get_prev_road(road_segment_id)
        upstream_lane_id = map_api._lane_by_address[(prev_road_id, lane_ordinal)]
    except NextRoadNotFound:
        upstream_lane_id = None
    try:
        next_road_id = map_model.get_next_road(road_segment_id)
        downstream_lane_id = map_api._lane_by_address[(next_road_id, lane_ordinal)]
    except NextRoadNotFound:
        downstream_lane_id = None
    return downstream_lane_id, upstream_lane_id


@pytest.fixture
def scene_static_no_split():
    MapService.initialize()
    return create_scene_static_from_map_api(MapService.get_instance())


@pytest.fixture
def scene_static():
    MapService.initialize('PG_split.bin')
    return create_scene_static_from_map_api(MapService.get_instance())


def create_scene_static_from_map_api(map_api: MapAPI,
                                     lane_modifications: Dict[int, List[Tuple[bool, int, int, float]]] = None) -> SceneStatic:
    if lane_modifications is None:
        lane_modifications = {}

    map_model = map_api._cached_map_model
    road_ids = map_model.get_road_ids()

    scene_road_segments = []
    for road_id in road_ids:
        num_lanes = map_model.get_road_data(road_id).lanes_num

        try:
            upstream_roads = np.array([map_model.get_prev_road(road_id)])
        except NextRoadNotFound:
            upstream_roads = np.array([])
        try:
            downstream_roads = np.array([map_model.get_next_road(road_id)])
        except NextRoadNotFound:
            downstream_roads = np.array([])
        lane_ids = np.array([map_api._lane_by_address[(road_id, i)] for i in range(num_lanes)])
        scene_road_segment = SceneRoadSegment(e_i_road_segment_id=road_id, e_i_road_id=0,
                                              e_Cnt_lane_segment_id_count=num_lanes,
                                              a_i_lane_segment_ids=lane_ids,
                                              e_e_road_segment_type=MapRoadSegmentType.Normal,
                                              e_Cnt_upstream_segment_count=len(upstream_roads),
                                              a_i_upstream_road_segment_ids=upstream_roads,
                                              e_Cnt_downstream_segment_count=len(downstream_roads),
                                              a_i_downstream_road_segment_ids=downstream_roads)

        scene_road_segments.append(scene_road_segment)
    scene_lane_segments_base = []
    scene_lane_segments_geo = []
    for lane_id in map_api._lane_address:

        road_segment_id, lane_ordinal = map_api._lane_address[lane_id]

        right_adj_lanes = [AdjacentLane(map_api._lane_by_address[(road_segment_id, k)],
                                        MovingDirection.Adjacent_or_same_dir,
                                        MapLaneType.LocalRoadLane) for k in range(lane_ordinal)]

        left_adj_lanes = [AdjacentLane(map_api._lane_by_address[(road_segment_id, k)],
                                       MovingDirection.Adjacent_or_same_dir,
                                       MapLaneType.LocalRoadLane) for k in range(lane_ordinal + 1,
                                                                                 map_model.get_road_data(
                                                                                     road_segment_id).lanes_num)]

        downstream_id, upstream_id = get_connectivity_lane_segment(map_api, road_segment_id, lane_ordinal)
        lane_frenet = map_api._lane_frenet[lane_id]
        nominal_points = []
        half_lane_width = map_api.get_road(road_segment_id).lane_width / 2
        for i in range(len(lane_frenet.O)):
            point = np.empty(len(list(NominalPathPoint)))
            point[NominalPathPoint.CeSYS_NominalPathPoint_e_l_EastX.value] = lane_frenet.O[i, FP_SX]
            point[NominalPathPoint.CeSYS_NominalPathPoint_e_l_NorthY.value] = lane_frenet.O[i, FP_DX]
            point[NominalPathPoint.CeSYS_NominalPathPoint_e_phi_heading.value] = np.arctan2(lane_frenet.T[i, 1],
                                                                                            lane_frenet.T[i, 0])
            point[NominalPathPoint.CeSYS_NominalPathPoint_e_il_curvature.value] = lane_frenet.k[i]
            point[NominalPathPoint.CeSYS_NominalPathPoint_e_il2_curvature_rate.value] = lane_frenet.k_tag[i]
            point[NominalPathPoint.CeSYS_NominalPathPoint_e_phi_cross_slope.value] = 0
            point[NominalPathPoint.CeSYS_NominalPathPoint_e_phi_along_slope.value] = 0
            point[NominalPathPoint.CeSYS_NominalPathPoint_e_l_s.value] = i * lane_frenet.ds
            point[NominalPathPoint.CeSYS_NominalPathPoint_e_l_left_offset.value] = half_lane_width
            point[NominalPathPoint.CeSYS_NominalPathPoint_e_l_right_offset.value] = -half_lane_width
            nominal_points.append(point)

        assert nominal_points[-1][NominalPathPoint.CeSYS_NominalPathPoint_e_l_s.value] == lane_frenet.s_max

        left_boundry_point = [BoundaryPoint(MapLaneMarkerType.MapLaneMarkerType_SolidSingleLine_BottsDots,
                                            0, lane_frenet.s_max)]

        right_boundry_point = [BoundaryPoint(MapLaneMarkerType.MapLaneMarkerType_SolidSingleLine_BottsDots,
                                             0, lane_frenet.s_max)]

        if not downstream_id:
            downstream_lane_segment_connectivity = []
        else:
            downstream_lane_segment_connectivity = [
                LaneSegmentConnectivity(downstream_id, ManeuverType.STRAIGHT_CONNECTION)]

        if not upstream_id:
            upstream_lane_segment_connectivity = []
        else:
            upstream_lane_segment_connectivity = [
                LaneSegmentConnectivity(upstream_id, ManeuverType.STRAIGHT_CONNECTION)]

        # Default lane attribute values
        num_active_lane_attributes = 4
        active_lane_attribute_indices = np.array([0, 1, 2, 3])
        lane_attributes = np.array([LaneMappingStatusType.CeSYS_e_LaneMappingStatusType_HDMap.value,
                                    GMAuthorityType.CeSYS_e_GMAuthorityType_None.value,
                                    LaneConstructionType.CeSYS_e_LaneConstructionType_Normal.value,
                                    MapLaneDirection.CeSYS_e_MapLaneDirection_SameAs_HostVehicle.value])
        lane_attribute_confidences = np.ones(4)

        # Check for lane attribute modifications
        if lane_id in lane_modifications:
            for lane_modification in lane_modifications[lane_id]:
                if lane_modification[0] is True:
                    lane_attributes[lane_modification[1]] = lane_modification[2]
                    lane_attribute_confidences[lane_modification[1]] = lane_modification[3]
                else:
                    active_lane_attribute_indices = np.delete(active_lane_attribute_indices, lane_modification[1])
                    num_active_lane_attributes -= 1

        scene_lane_segments_base.append(
            SceneLaneSegmentBase(e_i_lane_segment_id=lane_id,
                                 e_i_road_segment_id=road_segment_id,
                                 e_e_lane_type=MapLaneType.LocalRoadLane,
                                 e_Cnt_static_traffic_flow_control_count=0,
                                 as_static_traffic_flow_control=[],
                                 e_Cnt_dynamic_traffic_flow_control_count=0,
                                 as_dynamic_traffic_flow_control=[],
                                 e_Cnt_left_adjacent_lane_count=len(left_adj_lanes),
                                 as_left_adjacent_lanes=left_adj_lanes,
                                 e_Cnt_right_adjacent_lane_count=len(right_adj_lanes),
                                 as_right_adjacent_lanes=right_adj_lanes,
                                 e_Cnt_downstream_lane_count=len(
                                     downstream_lane_segment_connectivity),
                                 as_downstream_lanes=downstream_lane_segment_connectivity,
                                 e_Cnt_upstream_lane_count=len(upstream_lane_segment_connectivity),
                                 as_upstream_lanes=upstream_lane_segment_connectivity,
                                 e_v_nominal_speed=50.0,
                                 e_i_downstream_road_intersection_id=0,
                                 e_Cnt_lane_coupling_count=0,
                                 as_lane_coupling=[],
                                 e_Cnt_num_active_lane_attributes=num_active_lane_attributes,
                                 a_i_active_lane_attribute_indices=active_lane_attribute_indices,
                                 a_cmp_lane_attributes=lane_attributes,
                                 a_cmp_lane_attribute_confidences=lane_attribute_confidences)
        )

        scene_lane_segments_geo.append(
            SceneLaneSegmentGeometry(e_i_lane_segment_id=lane_id,
                                     e_i_road_segment_id=road_segment_id,
                                     e_Cnt_nominal_path_point_count=len(nominal_points),
                                     a_nominal_path_points=np.array(nominal_points),
                                     e_Cnt_left_boundary_points_count=len(left_boundry_point),
                                     as_left_boundary_points=left_boundry_point,
                                     e_Cnt_right_boundary_points_count=len(right_boundry_point),
                                     as_right_boundary_points=right_boundry_point)
        )

    header = Header(e_Cnt_SeqNum=0,
                    s_Timestamp=Timestamp(0, 0),
                    e_Cnt_version=0)

    map_origin = MapOrigin(e_phi_latitude=.0,
                           e_phi_longitude=.0,
                           e_l_altitude=.0,
                           s_Timestamp=Timestamp(0, 0))

    scene_road_intersections = []
    base_data = SceneStaticBase(e_Cnt_num_lane_segments=len(scene_lane_segments_base),
                                as_scene_lane_segments=scene_lane_segments_base,
                                e_Cnt_num_road_intersections=len(scene_road_intersections),
                                as_scene_road_intersection=scene_road_intersections,
                                e_Cnt_num_road_segments=len(scene_road_segments),
                                as_scene_road_segment=scene_road_segments)

    geometry_data = SceneStaticGeometry(e_Cnt_num_lane_segments=len(scene_lane_segments_geo),
                                        as_scene_lane_segments=scene_lane_segments_geo)

    road_segment_ids = [road_segment.e_i_road_segment_id for road_segment in scene_road_segments]

    nav_plan_data = NavigationPlan(e_Cnt_num_road_segments=len(scene_road_segments),
                                   a_i_road_segment_ids=np.array(road_segment_ids))

    return SceneStatic(s_Header=header,
                       s_Data=DataSceneStatic(e_b_Valid=True,
                                              s_RecvTimestamp=Timestamp(0, 0),
                                              e_l_perception_horizon_front=0.,
                                              e_l_perception_horizon_rear=0.,
                                              s_MapOrigin=map_origin,
                                              s_SceneStaticBase=base_data,
                                              s_SceneStaticGeometry=geometry_data,
                                              s_NavigationPlan=nav_plan_data))
