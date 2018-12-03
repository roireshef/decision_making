import pytest
import numpy as np

from decision_making.src.messages.scene_common_messages import Header, MapOrigin, Timestamp
from decision_making.src.messages.scene_static_message import SceneStatic, DataSceneStatic, SceneRoadSegment, \
    MapRoadSegmentType, SceneLaneSegment, MapLaneType, LaneSegmentConnectivity, ManeuverType, NominalPathPoint, \
    MapLaneMarkerType, BoundaryPoint, AdjacentLane, MovingDirection
from mapping.src.exceptions import NextRoadNotFound
from mapping.src.service.map_service import MapService


def get_connectivity_lane_segment(map_api, road_segment_id, lane_ordinal, lane_id):
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
def scene_static():
    MapService.initialize('PG_split.bin')
    map_api = MapService.get_instance()
    map_model = map_api._cached_map_model
    road_ids = map_model.get_road_ids()

    scene_road_segments = []
    for road_id in road_ids:
        num_lanes = map_model.get_road_data(road_id).lanes_num

        try:
            upstream_roads = np.array(map_model.get_prev_road(road_id))
        except NextRoadNotFound:
            upstream_roads = np.array([])
        try:
            downstream_roads = np.array(map_model.get_next_road(road_id))
        except NextRoadNotFound:
            downstream_roads = np.array([])
        lane_ids = np.array([map_api._lane_by_address[(road_id, i)] for i in range(num_lanes)])
        scene_road_segment = SceneRoadSegment(e_Cnt_road_segment_id=road_id, e_Cnt_road_id=0,
                                              e_Cnt_lane_segment_id_count=num_lanes,
                                              a_Cnt_lane_segment_id=lane_ids,
                                              e_e_road_segment_type=MapRoadSegmentType.Normal,
                                              e_Cnt_upstream_segment_count=1,
                                              a_Cnt_upstream_road_segment_id=upstream_roads,
                                              e_Cnt_downstream_segment_count=1,
                                              a_Cnt_downstream_road_segment_id=downstream_roads)

        scene_road_segments.append(scene_road_segment)
    scene_lane_segments = []
    for lane_id in map_api._lane_address:

        road_segment_id, lane_ordinal = map_api._lane_address[lane_id]

        right_adj_lanes = [AdjacentLane(map_api._lane_by_address[(road_segment_id, k)],
                                        MovingDirection.Adjacent_or_same_dir,
                                        MapLaneType.LocalRoadLane) for k in range(lane_ordinal)]

        left_adj_lanes = [AdjacentLane(map_api._lane_by_address[(road_segment_id, k)],
                                        MovingDirection.Adjacent_or_same_dir,
                                        MapLaneType.LocalRoadLane) for k in range(lane_ordinal+1,
                                                                                  map_model.get_road_data(road_id).lanes_num)]

        downstream_id, upstream_id = get_connectivity_lane_segment(map_api, road_segment_id, lane_ordinal, lane_id)
        lane_frenet = map_api._lane_frenet[lane_id]
        nominal_points = []
        half_lane_width = map_api.get_road(road_id).lane_width/2
        for i in range(len(lane_frenet.O)):
            point = np.empty(len(NominalPathPoint))
            point[NominalPathPoint.CeSYS_NominalPathPoint_e_l_EastX.value] = lane_frenet.O[i, 0]
            point[NominalPathPoint.CeSYS_NominalPathPoint_e_l_NorthY.value] = lane_frenet.O[i, 1]
            point[NominalPathPoint.CeSYS_NominalPathPoint_e_phi_heading.value] = np.arctan2(lane_frenet.T[i, 1],
                                                                                      lane_frenet.T[i, 0])
            point[NominalPathPoint.CeSYS_NominalPathPoint_e_il_curvature.value] = lane_frenet.k[i]
            point[NominalPathPoint.CeSYS_NominalPathPoint_e_il2_curvature_rate.value] = lane_frenet.k_tag[i]
            point[NominalPathPoint.CeSYS_NominalPathPoint_e_phi_cross_slope.value] = 0
            point[NominalPathPoint.CeSYS_NominalPathPoint_e_phi_along_slope.value] = 0
            point[NominalPathPoint.CeSYS_NominalPathPoint_e_l_s.value] = i * lane_frenet.ds
            point[NominalPathPoint.CeSYS_NominalPathPoint_e_l_left_offset.value] = half_lane_width
            point[NominalPathPoint.CeSYS_NominalPathPoint_e_l_right_offset.value] = half_lane_width
            nominal_points.append(point)

        left_boundry_point = [BoundaryPoint(MapLaneMarkerType.MapLaneMarkerType_SolidSingleLine_BottsDots,
                                                     0, lane_frenet.s_max)]

        right_boundry_point = [BoundaryPoint(MapLaneMarkerType.MapLaneMarkerType_SolidSingleLine_BottsDots,
                                           0, lane_frenet.s_max)]

        downstream_lane_segment_connectivity = LaneSegmentConnectivity(downstream_id, ManeuverType.STRAIGHT_CONNECTION)
        upstream_lane_segment_connectivity = LaneSegmentConnectivity(upstream_id, ManeuverType.STRAIGHT_CONNECTION)

        print('{3}:{0} {1} {2}'.format(downstream_id, lane_id, upstream_id, road_segment_id))
        scene_lane_segments.append(SceneLaneSegment(e_i_lane_segment_id=lane_id,
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
                                                    e_Cnt_downstream_lane_count=1,
                                                    as_downstream_lanes=[downstream_lane_segment_connectivity],
                                                    e_Cnt_upstream_lane_count=1,
                                                    as_upstream_lanes=[upstream_lane_segment_connectivity],
                                                    e_v_nominal_speed=50.0,
                                                    e_Cnt_nominal_path_point_count=len(nominal_points),
                                                    a_nominal_path_points=np.array(nominal_points),
                                                    e_Cnt_left_boundary_points_count=len(left_boundry_point),
                                                    as_left_boundary_points=left_boundry_point,
                                                    e_Cnt_right_boundary_points_count=len(right_boundry_point),
                                                    as_right_boundary_points=right_boundry_point,
                                                    e_i_downstream_road_intersection_id=0,
                                                    e_Cnt_lane_coupling_count=0,
                                                    as_lane_coupling=[]))

    header = Header(e_Cnt_SeqNum=0, s_Timestamp=Timestamp(0, 0),e_Cnt_version=0)
    map_origin = MapOrigin(e_phi_latitude=.0, e_phi_longitude=.0,e_l_altitude=.0,s_Timestamp=Timestamp(0,0))
    data = DataSceneStatic(e_b_Valid=True,
                           s_ComputeTimestamp=Timestamp(0, 0),
                           e_l_perception_horizon_front=.0,
                           e_l_perception_horizon_rear=.0,
                           e_Cnt_num_lane_segments=len(scene_lane_segments),
                           as_scene_lane_segment=scene_lane_segments,
                           e_Cnt_num_road_intersections=0,
                           as_scene_road_intersection=[],
                           e_Cnt_num_road_segments=len(scene_road_segments),
                           as_scene_road_segment=scene_road_segments)

    scene = SceneStatic(s_Header=header, s_MapOrigin=map_origin, s_Data=data)
    return scene


if __name__ == '__main__':
    ss = scene_static()

    #lanes = [lane.e_i_lane_segment_id for lane in ss.s_Data.as_scene_lane_segment]
    #print(lanes)



