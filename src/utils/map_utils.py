from typing import List, Dict

import numpy as np

from decision_making.src.global_constants import EPS
from decision_making.src.mapping.scene_model import SceneModel
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.messages.scene_static_message import NominalPathPoint, SceneLaneSegment, SceneRoadSegment
from decision_making.src.planning.behavioral.data_objects import RelativeLane
from decision_making.src.planning.types import CartesianPoint2D
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GeneralizedFrenetSerretFrame, \
    FrenetSubSegment
from decision_making.src.planning.utils.numpy_utils import NumpyUtils
from mapping.src.exceptions import raises, RoadNotFound, DownstreamLaneNotFound, \
    NavigationPlanTooShort, NavigationPlanDoesNotFitMap, AmbiguousNavigationPlan, UpstreamLaneNotFound, LaneNotFound


class MapUtils:


    @staticmethod
    def get_road_segment_ids() -> List[int]:
        """
        :return:road_segment_ids of every road in the static scene
        """
        scene_static = SceneModel.get_instance().get_scene_static()
        road_segments = scene_static.s_Data.as_scene_road_segment[:scene_static.s_Data.e_Cnt_num_road_segments]
        return [road_segment.e_Cnt_road_segment_id for road_segment in road_segments]

    @staticmethod
    def get_road_segment_id_from_lane_id(lane_id: int) -> int:
        """
        get road_segment_id containing the lane
        :param lane_id:
        :return: road_segment_id
        """
        lane = MapUtils.get_lane(lane_id)
        if lane is None:
            raise LaneNotFound('lane {0} cannot be found'.format(lane_id))
        return lane.e_i_road_segment_id

    @staticmethod
    def get_lane_ordinal(lane_id: int) -> int:
        """
        get lane ordinal of the lane on the road (the rightest lane's ordinal is 0)
        :param lane_id:
        :return: lane's ordinal
        """
        return MapUtils.get_lane(lane_id).e_Cnt_right_adjacent_lane_count

    @staticmethod
    def get_lane_length(lane_id: int) -> float:
        """
        get the whole lane's length
        :param lane_id:
        :return: lane's length
        """
        nominal_points = MapUtils.get_lane(lane_id).a_nominal_path_points
        return nominal_points[-1,NominalPathPoint.CeSYS_NominalPathPoint_e_l_s.value]

    @staticmethod
    def get_lane_frenet_frame(lane_id: int) -> FrenetSerret2DFrame:
        """
        get Frenet frame of the whole center-lane for the given lane
        :param lane_id:
        :return: Frenet frame
        """
        nominal_points = MapUtils.get_lane(lane_id).a_nominal_path_points
        points = nominal_points[:,(NominalPathPoint.CeSYS_NominalPathPoint_e_l_EastX.value,
                                   NominalPathPoint.CeSYS_NominalPathPoint_e_l_NorthY.value)]

        yaw = nominal_points[:, NominalPathPoint.CeSYS_NominalPathPoint_e_phi_heading.value]
        T = np.c_[np.cos(yaw), np.sin(yaw)]
        N = NumpyUtils.row_wise_normal(T)
        k = nominal_points[:, NominalPathPoint.CeSYS_NominalPathPoint_e_il_curvature.value][:, np.newaxis]
        k_tag = nominal_points[:, NominalPathPoint.CeSYS_NominalPathPoint_e_il2_curvature_rate.value][:, np.newaxis]
        ds = nominal_points[1, NominalPathPoint.CeSYS_NominalPathPoint_e_l_s.value]

        return FrenetSerret2DFrame(points=points, T=T, N=N, k=k, k_tag=k_tag, ds=ds)


    @staticmethod
    def get_adjacent_lanes(lane_id: int, relative_lane: RelativeLane) -> List[int]:
        """
        get sorted adjacent (right/left) lanes relative to the given lane segment, or empty list if no adjacent lanes
        :param lane_id:
        :param relative_lane: either right or left
        :return: adjacent lanes ids sorted by their distance from the given lane;
                    if there are no such lanes, return empty list []
        """

        lane = MapUtils.get_lane(lane_id)
        if relative_lane == RelativeLane.RIGHT_LANE:
            adj_lanes = lane.as_right_adjacent_lanes
        elif relative_lane == RelativeLane.LEFT_LANE:
            adj_lanes = lane.as_left_adjacent_lanes
        else:
            raise ValueError('Relative lane must be either right or left')
        return [l.e_Cnt_lane_segment_id for l in adj_lanes]

    @staticmethod
    def get_relative_lane_ids(lane_id: int) -> Dict[RelativeLane, int]:
        """
        get dictionary that given lane_id maps from RelativeLane to lane_id of the immediate neighbor lane
        :param lane_id:
        :return: dictionary from RelativeLane to the immediate neighbor lane ids (or None if the neighbor does not exist)
        """
        right_lanes = MapUtils.get_adjacent_lanes(lane_id, RelativeLane.RIGHT_LANE)
        left_lanes = MapUtils.get_adjacent_lanes(lane_id, RelativeLane.LEFT_LANE)
        return {RelativeLane.RIGHT_LANE: right_lanes[0] if len(right_lanes) > 0 else None,
                RelativeLane.SAME_LANE: lane_id,
                RelativeLane.LEFT_LANE: left_lanes[0] if len(left_lanes) > 0 else None}

    @staticmethod
    def get_closest_lane(cartesian_point: CartesianPoint2D, road_segment_id: int) -> int:
        """
        given cartesian coordinates, find the closest lane to the point
        :param cartesian_point: 2D cartesian coordinates
        :param road_segment_id: optional argument for road_segment_id closest to the given point
        :return: closest lane segment id
        """
        # TODO: This is a VERY naive implementation. It can be improved  by (a) vectorizing. i.e., all lanes stacked
        # TODO: (b) using current 's' to limit the search
        lane_ids = MapUtils.get_lanes_ids_from_road_segment_id(road_segment_id)
        min_dist = float('inf')
        min_lane_id = None
        for lane_id in lane_ids:
            nominal_points = MapUtils.get_lane(lane_id).a_nominal_path_points \
                [:, (NominalPathPoint.CeSYS_NominalPathPoint_e_l_EastX.value,
                    NominalPathPoint.CeSYS_NominalPathPoint_e_l_NorthY.value)]
            lane_min = np.min(np.linalg.norm(nominal_points-cartesian_point, axis=1))
            if lane_min < min_dist:
                min_dist = lane_min
                min_lane_id = lane_id
        return min_lane_id

    @staticmethod
    def get_dist_to_lane_borders(lane_id: int, s: float) -> (float, float):
        """
        get distance from the lane center to the lane borders at given longitude from the lane's origin
        :param lane_id:
        :param s: longitude of the lane center point (w.r.t. the lane Frenet frame)
        :return: distance from the right lane border, distance from the left lane border
        """
        nominal_points = MapUtils.get_lane(lane_id).a_nominal_path_points
        closest_s_idx = np.argmin(np.abs(nominal_points[:,
                                         NominalPathPoint.CeSYS_NominalPathPoint_e_l_s.value] - s))
        return (nominal_points[closest_s_idx, NominalPathPoint.CeSYS_NominalPathPoint_e_l_left_offset.value],
                nominal_points[closest_s_idx, NominalPathPoint.CeSYS_NominalPathPoint_e_l_right_offset.value])

    @staticmethod
    def get_dist_to_road_borders(lane_id: int, s: float) -> (float, float):
        """
         Get distance from the lane center to the road borders at given longitude from the lane's origin
        :param lane_id:
        :param s: longitude of the lane center point (w.r.t. the lane Frenet frame)
        :return: distance from the right road border, distance from the left road border
        """
        #TODO: Currently assuming that s is consistent across all lanes.

        right_lanes = MapUtils.get_adjacent_lanes(lane_id, RelativeLane.RIGHT_LANE)
        left_lanes = MapUtils.get_adjacent_lanes(lane_id, RelativeLane.LEFT_LANE)
        right_distance = np.sum([MapUtils.get_dist_to_lane_borders(right_lane, s) for right_lane in right_lanes])
        left_distance = np.sum([MapUtils.get_dist_to_lane_borders(left_lane, s) for left_lane in left_lanes])
        right_from_lane, left_from_lane = MapUtils.get_dist_to_lane_borders(lane_id, s)

        return right_from_lane+right_distance, left_from_lane+left_distance



    @staticmethod
    def get_lane_width(lane_id: int, s: float) -> float:
        """
        get lane width at given longitude from the lane's origin
        :param lane_id:
        :param s: longitude of the lane center point (w.r.t. the lane Frenet frame)
        :return: lane width
        """
        border_right, border_left = MapUtils.get_dist_to_lane_borders(lane_id, s)
        return border_right + border_left

    @staticmethod
    def get_upstream_lanes(lane_id: int) -> List[int]:
        """
        Get upstream lanes (incoming) of the given lane.
        This is referring only to the previous road-segment, and the returned list is there for many-to-1 connection.
        :param lane_id:
        :return: list of upstream lanes ids
        """
        upstream_connectivity = MapUtils.get_lane(lane_id).as_upstream_lanes
        return [connectivity.e_Cnt_lane_segment_id for connectivity in upstream_connectivity]

    @staticmethod
    def get_downstream_lanes(lane_id: int) -> List[int]:
        """
        Get downstream lanes (outgoing) of the given lane.
        This is referring only to the next road-segment, and the returned list is there for 1-to-many connection.
        :param lane_id:
        :return: list of downstream lanes ids
        """
        downstream_connectivity = MapUtils.get_lane(lane_id).as_downstream_lanes
        return [connectivity.e_Cnt_lane_segment_id for connectivity in downstream_connectivity]

    @staticmethod
    def get_lanes_ids_from_road_segment_id(road_segment_id: int) -> List[int]:
        """
        Get sorted list of lanes for given road segment. The output lanes are ordered by the lanes' ordinal,
        i.e. from the rightest lane to the most left.
        :param road_segment_id:
        :return: sorted list of lane segments' IDs
        """
        segment = MapUtils.get_road_segment(road_segment_id)
        return segment.a_Cnt_lane_segment_id.tolist()

    @staticmethod
    def does_map_exist_backward(lane_id: int, backward_dist: float):
        """
        check whether the map contains roads behind the given lane_id far enough (backward_dist)
        :param lane_id: current lane_id
        :param backward_dist: distance backward
        :return: True if the map contains upstream roads for the distance backward_dist
        """
        try:
            MapUtils._get_upstream_lanes_from_distance(lane_id, 0, backward_dist)
            return True
        except UpstreamLaneNotFound:
            return False

    @staticmethod
    def get_lookahead_frenet_frame(lane_id: int, starting_lon: float, lookahead_dist: float,
                                   navigation_plan: NavigationPlanMsg) -> GeneralizedFrenetSerretFrame:
        """
        Create Generalized Frenet frame of a given length along lane center, starting from given lane's longitude
        (may be negative).
        When some lane ends, it automatically continues to the next lane, according to the navigation plan.
        :param lane_id: starting lane_id
        :param starting_lon: starting longitude (may be negative) [m]
        :param lookahead_dist: lookahead distance for the output frame [m]
        :param navigation_plan: the relevant navigation plan to iterate over its road IDs.
        :return: generalized Frenet frame for the given route part
        """
        # find the starting point
        if starting_lon <= 0:  # the starting point is behind lane_id
            lane_ids, init_lon = MapUtils._get_upstream_lanes_from_distance(lane_id, 0, -starting_lon)
            init_lane_id = lane_ids[-1]
        else:  # the starting point is within or after lane_id
            init_lane_id, init_lon = lane_id, starting_lon

        # get the full lanes path
        sub_segments = MapUtils._advance_on_plan(init_lane_id, init_lon, lookahead_dist, navigation_plan)
        # create sub-segments for GFF
        frenet_frames = [MapUtils.get_lane_frenet_frame(sub_segment.segment_id) for sub_segment in sub_segments]
        # create GFF
        gff = GeneralizedFrenetSerretFrame.build(frenet_frames, sub_segments)
        return gff

    @staticmethod
    @raises(RoadNotFound, DownstreamLaneNotFound)
    def _advance_on_plan(initial_lane_id: int, initial_s: float, lookahead_distance: float,
                         navigation_plan: NavigationPlanMsg) -> List[FrenetSubSegment]:
        """
        Given a longitudinal position <initial_s> on lane segment <initial_lane_id>, advance <lookahead_distance>
        further according to <navigation_plan>, and finally return a configuration of lane-subsegments.
        If <desired_lon> is more than the distance to end of the plan, a LongitudeOutOfRoad exception is thrown.
        :param initial_lane_id: the initial lane_id (the vehicle is current on)
        :param initial_s: initial longitude along <initial_lane_id>
        :param lookahead_distance: the desired distance of lookahead in [m].
        :param navigation_plan: the relevant navigation plan to iterate over its road IDs.
        :return: a list of tuples of the format (lane_id, start_s (longitude) on lane, end_s (longitude) on lane)
        """
        initial_road_segment_id = MapUtils.get_road_segment_id_from_lane_id(initial_lane_id)
        initial_road_idx_on_plan = navigation_plan.get_road_index_in_plan(initial_road_segment_id)

        cumulative_distance = 0.
        lane_subsegments = []

        current_road_idx_on_plan = initial_road_idx_on_plan
        current_lane_id = initial_lane_id
        current_segment_start_s = initial_s  # reference longitudinal position on the lane of current_lane_id
        while True:
            current_lane_length = MapUtils.get_lane_length(current_lane_id)  # a lane's s_max

            # distance to travel on current lane: distance to end of lane, or shorter if reached <lookahead distance>
            current_segment_end_s = min(current_lane_length,
                                        current_segment_start_s + lookahead_distance - cumulative_distance)

            # add subsegment to the list and add traveled distance to <cumulative_distance> sum
            lane_subsegments.append(FrenetSubSegment(current_lane_id, current_segment_start_s, current_segment_end_s))
            cumulative_distance += current_segment_end_s - current_segment_start_s

            if cumulative_distance > lookahead_distance - EPS:
                break

            next_road_idx_on_plan = current_road_idx_on_plan + 1
            if next_road_idx_on_plan > len(navigation_plan.road_ids) - 1:
                raise NavigationPlanTooShort("Cannot progress further on plan %s (leftover: %s [m]); "
                                             "current_segment_end_s=%f lookahead_distance=%f" %
                                             (navigation_plan, lookahead_distance - cumulative_distance,
                                              current_segment_end_s, lookahead_distance))

            # pull next road segment from the navigation plan, then look for the downstream lane segment on this
            # road segment. This assumes a single correct downstream segment.
            next_road_segment_id_on_plan = navigation_plan.road_ids[next_road_idx_on_plan]
            downstream_lanes_ids = MapUtils.get_downstream_lanes(current_lane_id)

            if len(downstream_lanes_ids) == 0:
                raise DownstreamLaneNotFound("MapUtils._advance_on_plan: Downstream lane not found for lane_id=%d" % (current_lane_id))

            downstream_lanes_ids_on_plan = [lid for lid in downstream_lanes_ids
                                            if MapUtils.get_road_segment_id_from_lane_id(lid) == next_road_segment_id_on_plan]

            if len(downstream_lanes_ids_on_plan) == 0:
                raise NavigationPlanDoesNotFitMap("Any downstream lane is not in the navigation plan %s", (navigation_plan))
            if len(downstream_lanes_ids_on_plan) > 1:
                raise AmbiguousNavigationPlan("More than 1 downstream lanes according to the nav. plan %s", (navigation_plan))

            current_lane_id = downstream_lanes_ids_on_plan[0]
            current_segment_start_s = 0
            current_road_idx_on_plan = next_road_idx_on_plan

        return lane_subsegments

    @staticmethod
    def _get_upstream_lanes_from_distance(starting_lane_id: int, starting_lon: float, backward_dist: float) -> \
            (List[int], float):
        """
        given starting point (lane + starting_lon) on the lane and backward_dist, get list of lanes backward
        until reaching total distance from the starting point at least backward_dist
        :param starting_lane_id:
        :param starting_lon:
        :param backward_dist:
        :return: list of lanes backward and longitude on the last lane
        """
        path = [starting_lane_id]
        prev_lane_id = starting_lane_id
        total_dist = starting_lon
        while total_dist < backward_dist:
            prev_lane_ids = MapUtils.get_upstream_lanes(prev_lane_id)
            if len(prev_lane_ids) == 0:
                raise UpstreamLaneNotFound("MapUtils._advance_on_plan: Upstream lane not found for lane_id=%d" % (prev_lane_id))
            prev_lane_id = prev_lane_ids[0]
            path.append(prev_lane_id)
            total_dist += MapUtils.get_lane_length(prev_lane_id)
        return path, total_dist - backward_dist

    @staticmethod
    @raises(LaneNotFound)
    def get_lane(lane_id: int) -> SceneLaneSegment:
        """
        Retrieves lane by lane_id  according to the last message
        :param lane_id:
        :return:
        """
        scene_static = SceneModel.get_instance().get_scene_static()
        lanes = [lane for lane in scene_static.s_Data.as_scene_lane_segment if
                 lane.e_i_lane_segment_id == lane_id]
        if len(lanes) == 0:
            raise LaneNotFound('lane {0} not found'.format(lane_id))
        assert len(lanes) == 1
        return lanes[0]

    @staticmethod
    @raises(RoadNotFound)
    def get_road_segment(road_id: int) -> SceneRoadSegment:
        """

        Retrieves road by road_id  according to the last message
        :param road_id:
        :return:
        """
        scene_static = SceneModel.get_instance().get_scene_static()
        road_segments = [road_segment for road_segment in scene_static.s_Data.as_scene_road_segment if
                         road_segment.e_Cnt_road_segment_id == road_id]
        if len(road_segments) == 0 :
            raise RoadNotFound('road {0} not found '.format(road_id))
        assert len(road_segments) == 1
        return road_segments[0]

