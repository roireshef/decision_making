from decision_making.src.messages.scene_common_messages import Header, MapOrigin, Timestamp
from decision_making.src.messages.scene_static_message import SceneRoadSegment, MapRoadSegmentType, AdjacentLane, \
    MovingDirection, MapLaneType, DataSceneStatic, SceneStatic, SceneStaticBase, SceneStaticGeometry, NavigationPlan, \
    SceneLaneSegmentBase, SceneLaneSegmentGeometry, LaneSegmentConnectivity, ManeuverType, MapLaneMarkerType, BoundaryPoint, \
    LaneOverlap, LaneOverlapType
from decision_making.src.messages.scene_static_enums import NominalPathPoint, LaneMappingStatusType, MapLaneDirection, GMAuthorityType,\
    LaneConstructionType
from decision_making.src.planning.types import FP_SX, FP_DX
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from typing import List

import numpy as np


class SceneStaticUtils:

    @staticmethod
    def create_scene_static_from_points(road_segment_ids: List[int], num_lanes: int, lane_width: float,
                                        points_of_roads: List[np.array]) -> SceneStatic:
        """
        Create SceneStatic class based on the given lane-center points.
        :param road_segment_ids: list of road segments ids
        :param num_lanes: number of lanes on the road
        :param lane_width: lane width
        :param points_of_roads: list of arrays of road-center points per road segment
        :return: the instance of SceneStatic
        """
        lane_ids = []
        for road_segment_id in road_segment_ids:
            lane_ids.append(10 * road_segment_id + np.array(range(num_lanes)))

        scene_road_segments = []
        scene_lane_segments_base = []
        scene_lane_segments_geometry = []
        a_nominal_path_points = []

        for road_idx, road_segment_id in enumerate(road_segment_ids):
            downstream_roads = np.array([road_segment_ids[road_idx + 1]]) if road_idx < len(
                road_segment_ids) - 1 else np.array([])
            upstream_roads = np.array([road_segment_ids[road_idx - 1]]) if road_idx > 0 else np.array([])

            local_lane_ids = lane_ids[road_idx]
            scene_road_segment = SceneRoadSegment(e_i_road_segment_id=road_segment_id,
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

                lane_points = SceneStaticUtils._shift_road_points_laterally(points_of_roads[road_idx],
                                                                            (lane_ordinal - num_lanes/2 + 0.5) * lane_width)
                lane_frenet = FrenetSerret2DFrame.fit(lane_points)
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
                    downstream_lane_segments = []
                else:
                    downstream_lane_segments = [LaneSegmentConnectivity(downstream_id, ManeuverType.STRAIGHT_CONNECTION)]

                if not upstream_id:
                    upstream_lane_segments = []
                else:
                    upstream_lane_segments = [LaneSegmentConnectivity(upstream_id, ManeuverType.STRAIGHT_CONNECTION)]

                lane_overlap = [LaneOverlap(e_i_other_lane_segment_id=0,
                                            a_l_source_lane_overlap_stations=np.array([0, 0]),
                                            a_l_other_lane_overlap_stations=np.array([0, 0]),
                                            e_e_lane_overlap_type=LaneOverlapType.CeSYS_e_LaneOverlapType_Unknown)]

                scene_lane_segments_base.append(SceneLaneSegmentBase(e_i_lane_segment_id=lane_id,
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
                                                                     e_Cnt_downstream_lane_count=len(downstream_lane_segments),
                                                                     as_downstream_lanes=downstream_lane_segments,
                                                                     e_Cnt_upstream_lane_count=len(upstream_lane_segments),
                                                                     as_upstream_lanes=upstream_lane_segments,
                                                                     e_v_nominal_speed=50.0,
                                                                     e_l_length=nominal_points[-1][NominalPathPoint.CeSYS_NominalPathPoint_e_l_s.value],
                                                                     e_Cnt_num_active_lane_attributes=4,
                                                                     a_i_active_lane_attribute_indices=np.array([0, 1, 2, 3]),
                                                                     a_cmp_lane_attributes=
                                                                         [LaneMappingStatusType.CeSYS_e_LaneMappingStatusType_HDMap,
                                                                          GMAuthorityType.CeSYS_e_GMAuthorityType_None,
                                                                          LaneConstructionType.CeSYS_e_LaneConstructionType_Normal,
                                                                          MapLaneDirection.CeSYS_e_MapLaneDirection_SameAs_HostVehicle],
                                                                     a_cmp_lane_attribute_confidences=np.ones(4),
                                                                     e_Cnt_lane_overlap_count=0,
                                                                     as_lane_overlaps=lane_overlap))

                scene_lane_segments_geometry.append(SceneLaneSegmentGeometry(e_i_lane_segment_id=lane_id,
                                                                             e_i_road_segment_id=road_segment_id,
                                                                             e_Cnt_nominal_path_point_count=len(nominal_points),
                                                                             a_nominal_path_points=np.array(nominal_points),
                                                                             e_Cnt_left_boundary_points_count=len(left_boundry_point),
                                                                             as_left_boundary_points=left_boundry_point,
                                                                             e_Cnt_right_boundary_points_count=len(right_boundry_point),
                                                                             as_right_boundary_points=right_boundry_point))
                a_nominal_path_points.append(nominal_points)

        a_nominal_path_points = np.concatenate(a_nominal_path_points, axis=0)
        header = Header(e_Cnt_SeqNum=0, s_Timestamp=Timestamp(0, 0), e_Cnt_version=0)
        map_origin = MapOrigin(e_phi_latitude=.0, e_phi_longitude=.0, e_l_altitude=.0, s_Timestamp=Timestamp(0, 0))
        data = DataSceneStatic(e_b_Valid=True,
                               s_RecvTimestamp=Timestamp(0, 0),
                               e_l_perception_horizon_front=.0,
                               e_l_perception_horizon_rear=.0,
                               s_MapOrigin=map_origin,
                               s_SceneStaticBase=SceneStaticBase(e_Cnt_num_lane_segments=len(scene_lane_segments_base),
                                                                 as_scene_lane_segments=scene_lane_segments_base,
                                                                 e_Cnt_num_road_segments=len(scene_road_segments),
                                                                 as_scene_road_segment=scene_road_segments),
                               s_SceneStaticGeometry=SceneStaticGeometry(e_Cnt_num_lane_segments=len(scene_lane_segments_geometry),
                                                                         as_scene_lane_segments=scene_lane_segments_geometry,
                                                                         a_nominal_path_points=a_nominal_path_points),
                               s_NavigationPlan=NavigationPlan(e_Cnt_num_road_segments=len(road_segment_ids),
                                                               a_i_road_segment_ids=np.array(road_segment_ids)))

        scene = SceneStatic(s_Header=header, s_Data=data)
        return scene

    @staticmethod
    def _shift_road_points_laterally(points: np.array, lateral_shift: float) -> np.array:
        """
        Given points list along a road, shift them laterally by lat_shift [m]
        :param points (Nx2): points list along a given road
        :param lateral_shift: shift in meters
        :return: shifted points array (Nx2)
        """
        points_direction = np.diff(points, axis=0)
        norms = np.linalg.norm(points_direction, axis=1)[np.newaxis].T
        norms[np.where(norms == 0.0)] = 1.0
        direction_unit_vec = np.array(np.divide(points_direction, norms))
        normal_unit_vec = np.c_[-direction_unit_vec[:, 1], direction_unit_vec[:, 0]]
        normal_unit_vec = np.concatenate((normal_unit_vec, normal_unit_vec[-1, np.newaxis]))
        shifted_points = points + normal_unit_vec * lateral_shift
        return shifted_points


def test_shiftRoadPointsLaterally_simpleRoadShift1MRight_accurateShift():
    points = np.array([[0, 0], [1, -1], [1, -2]])
    shift = -1
    shifted_points = SceneStaticUtils._shift_road_points_laterally(points, shift)
    expected_shifted_points = np.array([[-1 / np.sqrt(2), -1 / np.sqrt(2)], [0, -1], [0, -2]])

    np.testing.assert_array_almost_equal(shifted_points, expected_shifted_points)
