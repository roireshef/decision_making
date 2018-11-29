from typing import List, Dict, Tuple, Optional

import numpy as np

from decision_making.src.global_constants import EPS
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.planning.behavioral.data_objects import RelativeLane
from decision_making.src.planning.types import FP_DX, FP_SX, C_X, C_Y, CartesianPoint2D
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from mapping.src.exceptions import raises, RoadNotFound, NextRoadNotFound, DownstreamLaneNotFound, \
    NavigationPlanTooShort, NavigationPlanDoesNotFitMap, AmbiguousNavigationPlan, UpstreamLaneNotFound
from mapping.src.service.map_service import MapService
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GeneralizedFrenetSerretFrame, \
    FrenetSubSegment


class MapUtils:
    @staticmethod
    def get_road_segment_id_from_lane_id(lane_id: int) -> int:
        """
        get road_segment_id containing the lane
        :param lane_id:
        :return: road_segment_id
        """
        return MapService.get_instance()._lane_address[lane_id][0]

    @staticmethod
    def get_lane_ordinal(lane_id: int) -> int:
        """
        get lane ordinal of the lane on the road (the rightest lane's ordinal is 0)
        :param lane_id:
        :return: lane's ordinal
        """
        return MapService.get_instance()._lane_address[lane_id][1]

    @staticmethod
    def get_lane_frenet_frame(lane_id: int) -> FrenetSerret2DFrame:
        """
        get Frenet frame of the whole center-lane for the given lane
        :param lane_id:
        :return: Frenet frame
        """
        return MapService.get_instance()._lane_frenet[lane_id]

    @staticmethod
    def get_lane_length(lane_id: int) -> float:
        """
        get the whole lane's length
        :param lane_id:
        :return: lane's length
        """
        return MapService.get_instance()._lane_frenet[lane_id].s_max

    @staticmethod
    def get_adjacent_lanes(lane_id: int, relative_lane: RelativeLane) -> List[int]:
        """
        get sorted adjacent (right/left) lanes relative to the given lane segment, or empty list if no adjacent lanes
        :param lane_id:
        :param relative_lane: either right or left
        :return: adjacent lanes ids sorted by their distance from the given lane;
                    if there are no such lanes, return empty list []
        """
        assert relative_lane != RelativeLane.SAME_LANE, "adjacent lanes can be either from LEFT or RIGHT side"
        map_api = MapService.get_instance()
        road_segment_id, lane_ordinal = map_api._lane_address[lane_id]
        num_lanes = map_api.get_num_lanes(road_segment_id)

        ordinals = {RelativeLane.RIGHT_LANE: range(lane_ordinal - 1, -1, -1),
                    RelativeLane.LEFT_LANE: range(lane_ordinal + 1, num_lanes)}[relative_lane]

        return [map_api._lane_by_address[(road_segment_id, ordinal)]
                for ordinal in ordinals if (road_segment_id, ordinal) in map_api._lane_by_address]

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

    # TODO: Remove it after introduction of the new mapping module. Avoid using this function once SP output is available.
    @staticmethod
    @raises(RoadNotFound)
    def get_closest_lane(cartesian_point: CartesianPoint2D, road_segment_id: int = None) -> int:
        """
        given cartesian coordinates, find the closest lane to the point
        :param cartesian_point: 2D cartesian coordinates
        :param road_segment_id: optional argument for road_segment_id closest to the given point
        :return: closest lane segment id
        """
        map_api = MapService.get_instance()
        relevant_road_ids = map_api._find_roads_containing_point(cartesian_point[C_X], cartesian_point[C_Y])
        if road_segment_id is None:
            # find the closest road segment
            closest_road_id = map_api._find_closest_road(cartesian_point[C_X], cartesian_point[C_Y], relevant_road_ids)
        else:
            if road_segment_id not in relevant_road_ids:
                raise RoadNotFound("road_segment_id=%d does not contain the given point", road_segment_id)
            closest_road_id = road_segment_id

        # find the closest lane segment, given the closest road segment
        num_lanes = map_api.get_road(closest_road_id).lanes_num
        try:
            # convert the given cpoint to fpoints w.r.t. to each lane's frenet frame
            fpoints = {}
            for lane_ordinal in range(num_lanes):
                lane_id = map_api._lane_by_address[(closest_road_id, lane_ordinal)]
                fpoints[lane_id] = map_api._lane_frenet[lane_id].cpoint_to_fpoint(cartesian_point)
            # find frenet points with minimal absolute latitude
            return min(fpoints.items(), key=lambda p: abs(p[1][FP_DX]))[0]
        except:
            return None

    @staticmethod
    def get_dist_to_lane_borders(lane_id: int, s: float) -> (float, float):
        """
        get distance from the lane center to the lane borders at given longitude from the lane's origin
        :param lane_id:
        :param s: longitude of the lane center point (w.r.t. the lane Frenet frame)
        :return: distance from the right lane border, distance from the left lane border
        """
        # this implementation assumes constant lane width (ignores the argument s)
        lane_width = MapService.get_instance().get_road(MapUtils.get_road_segment_id_from_lane_id(lane_id)).lane_width
        return lane_width / 2, lane_width / 2

    @staticmethod
    def get_dist_to_road_borders(lane_id: int, s: float) -> (float, float):
        """
        get distance from the lane center to the road borders at given longitude from the lane's origin
        :param lane_id:
        :param s: longitude of the lane center point (w.r.t. the lane Frenet frame)
        :return: distance from the right road border, distance from the left road border
        """
        # this implementation assumes constant lane width (ignores the argument s), the same width of all road's lanes
        map_api = MapService.get_instance()
        road_segment_id = MapUtils.get_road_segment_id_from_lane_id(lane_id)
        lane_width = map_api.get_road(road_segment_id).lane_width
        num_lanes = map_api.get_road(road_segment_id).lanes_num
        lane_ordinal = MapUtils.get_lane_ordinal(lane_id)
        return (lane_ordinal + 0.5) * lane_width, (num_lanes - lane_ordinal - 0.5) * lane_width

    @staticmethod
    def get_lane_width(lane_id: int, s: float) -> float:
        """
        get lane width at given longitude from the lane's origin
        :param lane_id:
        :param s: longitude of the lane center point (w.r.t. the lane Frenet frame)
        :return: lane width
        """
        dist_from_right, dist_from_left = MapUtils.get_dist_to_lane_borders(lane_id, s)
        return dist_from_right + dist_from_left

    @staticmethod
    def get_upstream_lanes(lane_id: int) -> List[int]:
        """
        Get upstream lanes (incoming) of the given lane.
        This is referring only to the previous road-segment, and the returned list is there for many-to-1 connection.
        :param lane_id:
        :return: list of upstream lanes ids
        """
        # TODO: use SP implementation, since this implementation assumes 1-to-1 lanes connectivity
        map_api = MapService.get_instance()
        try:
            prev_road_segment_id = map_api._cached_map_model.get_prev_road(
                MapUtils.get_road_segment_id_from_lane_id(lane_id))
            lane_ordinal = MapUtils.get_lane_ordinal(lane_id)
            return [map_api._lane_by_address[(prev_road_segment_id, lane_ordinal)]]
        except NextRoadNotFound:
            return []

    @staticmethod
    def get_downstream_lanes(lane_id: int) -> List[int]:
        """
        Get downstream lanes (outgoing) of the given lane.
        This is referring only to the next road-segment, and the returned list is there for 1-to-many connection.
        :param lane_id:
        :return: list of downstream lanes ids
        """
        # TODO: use SP implementation, since this implementation assumes 1-to-1 lanes connectivity
        map_api = MapService.get_instance()
        try:
            next_road_segment_id = map_api._cached_map_model.get_next_road(
                MapUtils.get_road_segment_id_from_lane_id(lane_id))
            lane_ordinal = MapUtils.get_lane_ordinal(lane_id)
            return [map_api._lane_by_address[(next_road_segment_id, lane_ordinal)]]
        except NextRoadNotFound:
            return []

    @staticmethod
    def get_lanes_ids_from_road_segment_id(road_segment_id: int) -> List[int]:
        """
        Get sorted list of lanes for given road segment. The output lanes are ordered by the lanes' ordinal,
        i.e. from the rightest lane to the most left.
        :param road_segment_id:
        :return: sorted list of lane segments' IDs
        """
        map_api = MapService.get_instance()
        num_lanes = map_api.get_road(road_segment_id).lanes_num
        lanes_list = []
        # get all lane segments for the given road segment
        for lane_ordinal in range(num_lanes):
            lane_id = map_api._lane_by_address[(road_segment_id, lane_ordinal)]
            lanes_list.append(lane_id)
        return lanes_list

    @staticmethod
    def get_lookahead_frenet_frame(lane_id: int, starting_lon: float, lookahead_dist: float,
                                   navigation_plan: NavigationPlanMsg) -> GeneralizedFrenetSerretFrame:
        """
        Create Generalized Frenet frame of a given length along lane center, starting from given lane's longitude
        (may be negative).
        When some lane finishes, it automatically continues to the next lane, according to the navigation plan.
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
