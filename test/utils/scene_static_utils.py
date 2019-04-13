from decision_making.src.messages.scene_common_messages import Header, MapOrigin, Timestamp
from decision_making.src.messages.scene_static_message import SceneRoadSegment, MapRoadSegmentType, AdjacentLane, \
    MovingDirection, MapLaneType, DataSceneStatic, SceneStatic, SceneLaneSegment, LaneSegmentConnectivity, ManeuverType, \
    MapLaneMarkerType, BoundaryPoint, NominalPathPoint
from decision_making.src.planning.types import FP_SX, FP_DX
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from typing import List

import numpy as np


class SceneStaticUtils:

    @staticmethod
    def create_scene_static_from_points(road_segment_ids: List[int], num_lanes: int, lane_width: float,
                                        points_of_roads: List[np.array]) -> SceneStatic:
        lane_ids = []
        for road_segment_id in road_segment_ids:
            lane_ids.append(10 * road_segment_id + np.array(range(num_lanes)))

        scene_road_segments = []
        scene_lane_segments = []
        for road_idx, road_segment_id in enumerate(road_segment_ids):
            upstream_roads = np.array([road_segment_ids[road_idx + 1]]) if road_idx < len(
                road_segment_ids) - 1 else np.array([])
            downstream_roads = np.array([road_segment_ids[road_idx - 1]]) if road_idx > 0 else np.array([])

            local_lane_ids = lane_ids[road_idx]
            scene_road_segment = SceneRoadSegment(e_i_road_segment_id=road_segment_id, e_i_road_id=0,
                                                  e_Cnt_lane_segment_id_count=num_lanes,
                                                  a_i_lane_segment_ids=local_lane_ids,
                                                  e_e_road_segment_type=MapRoadSegmentType.Normal,
                                                  e_Cnt_upstream_segment_count=len(upstream_roads),
                                                  a_i_upstream_road_segment_ids=upstream_roads,
                                                  e_Cnt_downstream_segment_count=len(downstream_roads),
                                                  a_i_downstream_road_segment_ids=downstream_roads)

            scene_road_segments.append(scene_road_segment)

            for lane_ordinal, lane_id in enumerate(lane_ids[road_idx]):

                right_adj_lanes = [AdjacentLane(local_lane_ids[k],
                                                MovingDirection.Adjacent_or_same_dir, MapLaneType.LocalRoadLane)
                                   for k in range(lane_ordinal - 1, -1, -1)]

                left_adj_lanes = [AdjacentLane(local_lane_ids[k],
                                               MovingDirection.Adjacent_or_same_dir, MapLaneType.LocalRoadLane)
                                  for k in range(lane_ordinal + 1, num_lanes)]

                downstream_id = lane_ids[road_idx + 1][lane_ordinal] if road_idx < len(road_segment_ids) - 1 else None
                upstream_id = lane_ids[road_idx - 1][lane_ordinal] if road_idx > 0 else None

                lane_frenet = FrenetSerret2DFrame.fit(points_of_roads[road_idx])
                nominal_points = []
                half_lane_width = lane_width / 2
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
                    downstream_lane_segment_connectivity = [LaneSegmentConnectivity(downstream_id, ManeuverType.STRAIGHT_CONNECTION)]

                if not upstream_id:
                    upstream_lane_segment_connectivity = []
                else:
                    upstream_lane_segment_connectivity = [LaneSegmentConnectivity(upstream_id, ManeuverType.STRAIGHT_CONNECTION)]

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
                                                            e_Cnt_downstream_lane_count=len(
                                                                downstream_lane_segment_connectivity),
                                                            as_downstream_lanes=downstream_lane_segment_connectivity,
                                                            e_Cnt_upstream_lane_count=len(
                                                                upstream_lane_segment_connectivity),
                                                            as_upstream_lanes=upstream_lane_segment_connectivity,
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

        header = Header(e_Cnt_SeqNum=0, s_Timestamp=Timestamp(0, 0), e_Cnt_version=0)
        map_origin = MapOrigin(e_phi_latitude=.0, e_phi_longitude=.0, e_l_altitude=.0, s_Timestamp=Timestamp(0, 0))
        scene_road_intersections = []
        data = DataSceneStatic(e_b_Valid=True,
                               s_RecvTimestamp=Timestamp(0, 0),
                               s_ComputeTimestamp=Timestamp(0, 0),
                               e_l_perception_horizon_front=.0,
                               e_l_perception_horizon_rear=.0,
                               e_Cnt_num_lane_segments=len(scene_lane_segments),
                               as_scene_lane_segment=scene_lane_segments,
                               e_Cnt_num_road_intersections=len(scene_road_intersections),
                               as_scene_road_intersection=scene_road_intersections,
                               e_Cnt_num_road_segments=len(scene_road_segments),
                               as_scene_road_segment=scene_road_segments)

        scene = SceneStatic(s_Header=header, s_MapOrigin=map_origin, s_Data=data)
        return scene
