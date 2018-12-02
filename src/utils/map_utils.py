from typing import List

import itertools
import numpy as np

from decision_making.src.mapping.scene_model import SceneModel
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.messages.scene_static_message import NominalPathPoint
from decision_making.src.planning.behavioral.data_objects import RelativeLane
from decision_making.src.planning.types import CartesianPoint2D, FS_SX
from decision_making.src.planning.types import FS_DX
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.planning.utils.generalized_frenet_serret_frame import FrenetSubSegment, GeneralizedFrenetSerretFrame
from decision_making.src.state.map_state import MapState
from mapping.src.model.constants import ROAD_SHOULDERS_WIDTH


class MapUtils:


    # TODO: remove this on Lane-based planner PR
    @staticmethod
    def get_road_rhs_frenet(road_id: int) -> FrenetSerret2DFrame:
        rhs_lane_id = MapUtils.get_adjacent_lanes(road_id, RelativeLane.RIGHT_LANE)[0]
        nominal_points = SceneModel.get_instance().get_lane(rhs_lane_id).a_nominal_path_points
        return FrenetSerret2DFrame.fit(nominal_points[:,
                                       (NominalPathPoint.CeSYS_NominalPathPoint_e_l_right_offset,
                                        NominalPathPoint.CeSYS_NominalPathPoint_e_l_NorthY)])



    @staticmethod
    def get_lookahead_frenet_frame(lane_id: int, starting_lon: float, lookahead_dist: float,
                                   navigation_plan: NavigationPlanMsg) -> GeneralizedFrenetSerretFrame:
        """
        Get Frenet frame of a given length along lane center, starting from given lane's longitude (may be negative).
        When some lane finishes, it automatically continues to the next lane, according to the navigation plan.
        :param lane_id: starting lane_id
        :param starting_lon: starting longitude (may be negative) [m]
        :param lookahead_dist: lookahead distance for the output frame [m]
        :param navigation_plan: the relevant navigation plan to iterate over its road IDs.
        :return: Frenet frame for the given route part
        """
        scene_static = SceneModel.get_instance().get_scene_static()

        # TODO CHECK can SceneModel give us a larger horizon than s_Data.e_l_perception_horizon_*
        if (starting_lon + lookahead_dist) > scene_static.s_Data.e_l_perception_horizon_front:
            raise ValueError('lookahead_dist greater than SceneStatic front horizon')
        if abs(starting_lon) > scene_static.s_Data.e_l_perception_horizon_rear:   
            raise ValueError('starting_lon greater than SceneStatic rear horizon')
        
        nav_plan_laneseg_ids = [MapUtils.get_lanes_id_from_road_segment_id(road_id) for road_id in navigation_plan.road_ids]
        nav_plan_laneseg_ids = list(itertools.chain.from_iterable(nav_plan_laneseg_ids))

        # TODO CHECK can scenemodel have function to give list of all available lane segments
        scene_model_lane_seg_ids = [lane.e_i_lane_segment_id for lane in scene_static.s_Data.as_scene_lane_segment]

        if not set(nav_plan_laneseg_ids).issubset(scene_model_lane_seg_ids):
            raise ValueError('Navigation plan includes lane IDs that are not part of SceneModel')

        # TODO ASSUMPTION starting_lon is limited to current lane id and previous, not >current <prev
        start_lane_seg_id = lane_id if starting_lon >= 0 else SceneModel.get_instance().get_lane(lane_id).as_upstream_lanes[0].e_Cnt_lane_segment_id 
        start_lane = SceneModel.get_instance().get_lane(start_lane_seg_id)
        
        # Find start point s distance
        if starting_lon < 0:     
            seg_start_s = start_lane.a_nominal_path_points[start_lane.e_Cnt_nominal_path_point_count][NominalPathPoint.CeSYS_NominalPathPoint_e_l_s.value] + starting_lon
        else:
            seg_start_s = starting_lon 

        # Find start point s index
        start_pt_idx = 0
        while start_lane.a_nominal_path_points[start_pt_idx][NominalPathPoint.CeSYS_NominalPathPoint_e_l_s.value] < seg_start_s:
            start_pt_idx += 1

        accumulated_s = 0
        frenet_frames = []
        frenet_sub_segments = [] 
        curr_lane_id = start_lane_seg_id
        while accumulated_s < lookahead_dist:
            curr_lane = SceneModel.get_instance().get_lane(curr_lane_id) 
            first_nom_path_pt = curr_lane.a_nominal_path_points[0]
            last_nom_path_pt = curr_lane.a_nominal_path_points[curr_lane.e_Cnt_nominal_path_point_count - 1]

            # End point is beyond end of LS and LS is starting segment. Append start point to end of LS 
            if curr_lane_id == start_lane_seg_id and \
               (last_nom_path_pt[NominalPathPoint.CeSYS_NominalPathPoint_e_l_s.value] - curr_lane.a_nominal_path_points[start_pt_idx][NominalPathPoint.CeSYS_NominalPathPoint_e_l_s.value]) < lookahead_dist:
                accumulated_s += last_nom_path_pt[NominalPathPoint.CeSYS_NominalPathPoint_e_l_s.value] - curr_lane.a_nominal_path_points[start_pt_idx][NominalPathPoint.CeSYS_NominalPathPoint_e_l_s.value]
                frenet_frames += FrenetSerret2DFrame.fit(curr_lane.a_nominal_path_points[0:curr_lane.e_Cnt_nominal_path_point_count, 
                                                         NominalPathPoint.CeSYS_NominalPathPoint_e_l_EastX.value:NominalPathPoint.CeSYS_NominalPathPoint_e_l_NorthY.value]) 
 
                frenet_sub_segments += FrenetSubSegment(curr_lane.e_i_lane_segment_id,
                                                        curr_lane.a_nominal_path_points[start_pt_idx][NominalPathPoint.CeSYS_NominalPathPoint_e_l_s.value],
                                                        last_nom_path_pt[NominalPathPoint.CeSYS_NominalPathPoint_e_l_s.value],
                                                        frenet_frames[-1].ds)
                curr_lane_id = curr_lane.as_downstream_lanes[0].e_Cnt_lane_segment_id
            
            # End point beyond end of LS. Append entire LS 
            elif accumulated_s + last_nom_path_pt[NominalPathPoint.CeSYS_NominalPathPoint_e_l_s.value] < lookahead_dist:
                accumulated_s += last_nom_path_pt[NominalPathPoint.CeSYS_NominalPathPoint_e_l_s.value]
                
                frenet_frames += MapUtils.get_lane_frenet_frame(curr_lane.e_i_lane_segment_id)
                frenet_sub_segments += FrenetSubSegment(curr_lane.e_i_lane_segment_id,
                                                        first_nom_path_pt[NominalPathPoint.CeSYS_NominalPathPoint_e_l_s.value],
                                                        last_nom_path_pt[NominalPathPoint.CeSYS_NominalPathPoint_e_l_s.value],
                                                        frenet_frames[-1].ds)
                curr_lane_id = curr_lane.as_downstream_lanes[0].e_Cnt_lane_segment_id

            # End point is somewhere in the middle of LS. Find the endpoint and append the relevant portion of LS
            else:
                pt_idx = 0
                while (accumulated_s + curr_lane.a_nominal_path_points[pt_idx][NominalPathPoint.CeSYS_NominalPathPoint_e_l_s.value]) < lookahead_dist:
                    pt_idx += 1

                accumulated_s += curr_lane.a_nominal_path_points[pt_idx][NominalPathPoint.CeSYS_NominalPathPoint_e_l_s.value]
                frenet_frames += FrenetSerret2DFrame.fit(curr_lane.a_nominal_path_points[0:pt_idx, 
                                                         NominalPathPoint.CeSYS_NominalPathPoint_e_l_EastX.value:NominalPathPoint.CeSYS_NominalPathPoint_e_l_NorthY.value]) 
                if curr_lane_id == start_lane_seg_id:
                    frenet_sub_segments += FrenetSubSegment(curr_lane.e_i_lane_segment_id,
                                                            curr_lane.a_nominal_path_points[start_pt_idx][NominalPathPoint.CeSYS_NominalPathPoint_e_l_s.value],
                                                            curr_lane.a_nominal_path_points[pt_idx][NominalPathPoint.CeSYS_NominalPathPoint_e_l_s.value],
                                                            frenet_frames[-1].ds)
                else:
                    frenet_sub_segments += FrenetSubSegment(curr_lane.e_i_lane_segment_id,
                                                            first_nom_path_pt[NominalPathPoint.CeSYS_NominalPathPoint_e_l_s.value],
                                                            curr_lane.a_nominal_path_points[pt_idx][NominalPathPoint.CeSYS_NominalPathPoint_e_l_s.value],
                                                            frenet_frames[-1].ds)

        return GeneralizedFrenetSerretFrame.build(frenet_frames, frenet_sub_segments)

    # TODO: remove this on Lane-based planner PR
    # TODO: Note! This function is only valid when the frenet reference frame is from the right side of the road
    @staticmethod
    def is_object_on_road(map_state):
        # type: (MapState) -> bool
        """
        Returns true of the object is on the road. False otherwise.
        Note! This function is valid only when the frenet reference frame is from the right side of the road
        :param map_state: the map state to check
        :return: Returns true of the object is on the road. False otherwise.
        """

        current_s = map_state.road_fstate[FS_SX]
        road_width = np.sum([MapUtils.get_lane_width(lane_id, current_s)
                             for lane_id in MapUtils.get_lanes_id_from_road_segment_id(map_state.road_id)])
        is_on_road = road_width + ROAD_SHOULDERS_WIDTH > map_state.road_fstate[FS_DX] > -ROAD_SHOULDERS_WIDTH
        return is_on_road



    # TODO: remove it after introduction of the new mapping module
    @staticmethod
    def get_closest_lane(cartesian_point: CartesianPoint2D, road_segment_id: int = None) -> int:
        """
        given cartesian coordinates, find the closest lane to the point
        :param cartesian_point: 2D cartesian coordinates
        :param road_segment_id: optional argument for road_segment_id closest to the given point
        :return: closest lane segment id
        """
        # TODO: This is a VERY naive implementation. This can be imporoved easily by (a) vectorizing. i.e., all lanes stacked
        # TODO: (b) using current 's' to limit the search
        lane_ids = MapUtils.get_lanes_id_from_road_segment_id(road_segment_id)
        min_dist = float('inf')
        min_lane_id = None
        for lane_id in lane_ids:
            nominal_points = SceneModel.get_instance() \
                                 .get_lane(lane_id).a_nominal_path_points \
                [:, (NominalPathPoint.CeSYS_NominalPathPoint_e_l_EastX,
                    NominalPathPoint.CeSYS_NominalPathPoint_e_l_NorthY)]
            lane_min = np.min(np.linalg.norm(nominal_points-cartesian_point, axis=1))
            if lane_min < min_dist:
                min_dist = lane_min
                min_lane_id = lane_id
        return min_lane_id


    @staticmethod
    def get_road_segment_id_from_lane_id(lane_id: int) -> int:
        """
        get road_segment_id containing the lane
        :param lane_id:
        :return: road_segment_id
        """
        return SceneModel.get_instance().get_lane(lane_id).e_i_road_segment_id

    @staticmethod
    def get_lane_ordinal(lane_id: int) -> int:
        """
        get lane ordinal of the lane on the road (the rightest lane's ordinal is 0)
        :param lane_id:
        :return: lane's ordinal
        """
        return SceneModel.get_instance().get_lane(lane_id).e_Cnt_right_adjacent_lane_count


    @staticmethod
    def get_lane_length(lane_id: int) -> float:
        """
        get the whole lane's length
        :param lane_id:
        :return: lane's length
        """
        nominal_points = SceneModel.get_instance().get_lane(lane_id).a_nominal_path_points
        return nominal_points[-1][NominalPathPoint.CeSYS_NominalPathPoint_e_l_s.value]


    @staticmethod
    def get_lane_frenet_frame(lane_id: int) -> FrenetSerret2DFrame:
        """
        get Frenet frame of the whole center-lane for the given lane
        :param lane_id:
        :return: Frenet frame
        """
        nominal_points = SceneModel.get_instance().get_lane(lane_id).a_nominal_path_points
        return FrenetSerret2DFrame.fit(nominal_points[:,
                                       (NominalPathPoint.CeSYS_NominalPathPoint_e_l_EastX,
                                        NominalPathPoint.CeSYS_NominalPathPoint_e_l_NorthY)])


    @staticmethod
    def get_adjacent_lanes(lane_id: int, relative_lane: RelativeLane) -> List[int]:
        """
        get sorted adjacent (right/left) lanes relative to the given lane segment
        :param lane_id:
        :param relative_lane: either right or left
        :return: adjacent lanes ids sorted by their distance from the given lane
        """
        lane = SceneModel.get_instance().get_lane(lane_id)
        if relative_lane == RelativeLane.RIGHT_LANE:
            adj_lanes = lane.as_right_adjacent_lanes
        elif relative_lane == RelativeLane.LEFT_LANE:
            adj_lanes = lane.as_left_adjacent_lanes
        else:
            raise ValueError('Relative lane must be either right or left')
        return [l.e_Cnt_lane_segment_id for l in adj_lanes]



    @staticmethod
    def get_dist_from_lane_center_to_lane_borders(lane_id: int, s: float) -> (float, float):
        """
        get distance from the lane center to the lane borders at given longitude from the lane's origin
        :param lane_id:
        :param s: longitude of the lane center point (w.r.t. the lane Frenet frame)
        :return: distance from the right lane border, distance from the left lane border
        """
        nominal_points = SceneModel.get_instance().get_lane(lane_id).a_nominal_path_points
        for x in nominal_points:
            if x[NominalPathPoint.CeSYS_NominalPathPoint_e_l_s.value] == s:
                return (x[NominalPathPoint.CeSYS_NominalPathPoint_e_l_left_offset.value],
                        x[NominalPathPoint.CeSYS_NominalPathPoint_e_l_right_offset.value])


    @staticmethod
    def get_dist_from_lane_center_to_road_borders(lane_id: int, s: float) -> (float, float):
        """
        get distance from the lane center to the road borders at given longitude from the lane's origin
        :param lane_id:
        :param s: longitude of the lane center point (w.r.t. the lane Frenet frame)
        :return: distance from the right road border, distance from the left road border
        """
        right_lanes = MapUtils.get_adjacent_lanes(lane_id, RelativeLane.RIGHT_LANE)
        if len(right_lanes) > 0:
            rightmost_lane = right_lanes[0]
        else:
            rightmost_lane = lane_id

        right_lanes = MapUtils.get_adjacent_lanes(lane_id, RelativeLane.LEFT_LANE)
        if len(right_lanes) > 0:
            leftmost_lane = right_lanes[-1]
        else:
            leftmost_lane = lane_id
        right_border, _ = MapUtils.get_dist_from_lane_center_to_lane_borders(rightmost_lane, s)
        _, left_border = MapUtils.get_dist_from_lane_center_to_lane_borders(leftmost_lane, s)
        return right_border, left_border



    @staticmethod
    def get_lane_width(lane_id: int, s: float) -> float:
        """
        get lane width at given longitude from the lane's origin
        :param lane_id:
        :param s: longitude of the lane center point (w.r.t. the lane Frenet frame)
        :return: lane width
        """
        return np.sum(np.abs(MapUtils.get_dist_from_lane_center_to_lane_borders(lane_id, s)))



    @staticmethod
    def get_upstream_lanes(lane_id: int) -> List[int]:
        """
        get upstream lanes (incoming) of the given lane
        :param lane_id:
        :return: list of upstream lanes ids
        """
        upstream_connectivity = SceneModel.get_instance().get_lane(lane_id).as_upstream_lanes
        return [connectivity.e_Cnt_lane_segment_id for connectivity in upstream_connectivity]


    @staticmethod
    def get_downstream_lanes(lane_id: int) -> List[int]:
        """
        get downstream lanes (outgoing) of the given lane
        :param lane_id:
        :return: list of downstream lanes ids
        """
        downstream_connectivity = SceneModel.get_instance().get_lane(lane_id).as_downstream_lanes
        return [connectivity.e_Cnt_lane_segment_id for connectivity in downstream_connectivity]

    @staticmethod
    def get_lanes_id_from_road_segment_id(road_segment_id: int) -> List[int]:
        """
        Get sorted list of lanes for given road segment. The output lanes are ordered by the lanes' ordinal,
        i.e. from the rightest lane to the most left.
        :param road_segment_id:
        :return: sorted list of lane segments' IDs
        """
        return SceneModel.get_instance().get_road_segment(road_segment_id).a_Cnt_lane_segment_id
