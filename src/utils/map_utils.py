from typing import List, Optional

import numpy as np

from decision_making.src.global_constants import EPS
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.planning.behavioral.data_objects import RelativeLane
from decision_making.src.planning.types import FP_DX, FP_SX, C_X, C_Y, CartesianPoint2D, FrenetPoint
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from mapping.src.exceptions import raises, LongitudeOutOfRoad, RoadNotFound, UpstreamLaneNotFound, \
    DownstreamLaneNotFound, NextRoadNotFound
from mapping.src.service.map_service import MapService


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
        get sorted adjacent (right/left) lanes relative to the given lane segment
        :param lane_id:
        :param relative_lane: either right or left
        :return: adjacent lanes ids sorted by their distance from the given lane
        """
        assert relative_lane != RelativeLane.SAME_LANE
        map_api = MapService.get_instance()
        road_segment_id, lane_ordinal = map_api._lane_address[lane_id]
        num_lanes = map_api.get_num_lanes(road_segment_id)

        ordinals = {RelativeLane.RIGHT_LANE: range(lane_ordinal - 1, -1, -1),
                    RelativeLane.LEFT_LANE: range(lane_ordinal + 1, num_lanes)}[relative_lane]

        return [map_api._lane_by_address[(road_segment_id, ordinal)]
                for ordinal in ordinals if (road_segment_id, ordinal) in map_api._lane_by_address]

    # TODO: remove it after introduction of the new mapping module
    @staticmethod
    def get_closest_lane(cartesian_point: CartesianPoint2D, road_segment_id: int=None) -> int:
        """
        given cartesian coordinates, find the closest lane to the point
        :param cartesian_point: 2D cartesian coordinates
        :param road_segment_id: optional argument for road_segment_id closest to the given point
        :return: closest lane segment id
        """
        map_api = MapService.get_instance()
        if road_segment_id is None:
            # find the closest road segment
            relevant_road_ids = map_api._find_roads_containing_point(cartesian_point[C_X], cartesian_point[C_Y])
            closest_road_id = map_api._find_closest_road(cartesian_point[C_X], cartesian_point[C_Y], relevant_road_ids)
        else:
            closest_road_id = road_segment_id

        # find the closest lane segment, given the closest road segment
        num_lanes = map_api.get_road(closest_road_id).lanes_num
        # convert the given cpoint to fpoints w.r.t. to each lane's frenet frame
        fpoints = {}
        for lane_ordinal in range(num_lanes):
            lane_id = map_api._lane_by_address[(closest_road_id, lane_ordinal)]
            fpoints[lane_id] = map_api._lane_frenet[lane_id].cpoint_to_fpoint(cartesian_point)
        # find frenet points with minimal absolute latitude
        return min(fpoints.items(), key=lambda p: abs(p[1][FP_DX]))[0]

    @staticmethod
    def get_dist_from_lane_center_to_lane_borders(lane_id: int, s: float) -> (float, float):
        """
        get distance from the lane center to the lane borders at given longitude from the lane's origin
        :param lane_id:
        :param s: longitude of the lane center point (w.r.t. the lane Frenet frame)
        :return: distance from the right lane border, distance from the left lane border
        """
        # this implementation assumes constant lane width (ignores the argument s)
        lane_width = MapService.get_instance().get_road(MapUtils.get_road_segment_id_from_lane_id(lane_id)).lane_width
        return lane_width/2, lane_width/2

    @staticmethod
    def get_dist_from_lane_center_to_road_borders(lane_id: int, s: float) -> (float, float):
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
        return (lane_ordinal + 0.5)*lane_width, (num_lanes - lane_ordinal - 0.5)*lane_width

    @staticmethod
    def get_lane_width(lane_id: int, s: float) -> float:
        """
        get lane width at given longitude from the lane's origin
        :param lane_id:
        :param s: longitude of the lane center point (w.r.t. the lane Frenet frame)
        :return: lane width
        """
        dist_from_right, dist_from_left = MapUtils.get_dist_from_lane_center_to_lane_borders(lane_id, s)
        return dist_from_right + dist_from_left

    @staticmethod
    def get_lookahead_frenet_frame(lane_id: int, starting_lon: float, lookahead_dist: float,
                                   navigation_plan: NavigationPlanMsg):
        """
        Get Frenet frame of a given length along lane center, starting from given lane's longitude (may be negative).
        When some lane finishes, it automatically continues to the next lane, according to the navigation plan.
        :param lane_id: starting lane_id
        :param starting_lon: starting longitude (may be negative) [m]
        :param lookahead_dist: lookahead distance for the output frame [m]
        :param navigation_plan: the relevant navigation plan to iterate over its road IDs.
        :return: Frenet frame for the given route part
        """
        # TODO: deal with starting_lon < 0, when get_upstream_lanes() is implemented
        # in current implementation: if starting_lon < 0, extract Frenet frame with only positive longitudes
        if starting_lon < EPS:
            lookahead_dist += starting_lon
            starting_lon = EPS

        points, _ = MapUtils._get_lookahead_points(lane_id, starting_lon, lookahead_dist, desired_lat=0,
                                                   navigation_plan=navigation_plan)

        center_lane_reference_route = FrenetSerret2DFrame.fit(points)
        return center_lane_reference_route

    @staticmethod
    def get_upstream_lanes(lane_id: int) -> List[int]:
        """
        get upstream lanes (incoming) of the given lane
        :param lane_id:
        :return: list of upstream lanes ids
        """
        pass

    @staticmethod
    def get_downstream_lanes(lane_id: int) -> List[int]:
        """
        get downstream lanes (outgoing) of the given lane
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
    def _convert_from_lane_to_map_coordinates(lane_id: int, frenet_point: FrenetPoint, relative_yaw: float=0) -> \
            [CartesianPoint2D, float]:
        """
        convert a point from lane coordinates to map (global) coordinates
        :param lane_id:
        :param frenet_point: frenet point w.r.t. the lane
        :param relative_yaw: intra-lane yaw (optional)
        :return: map coordinates: x, y, yaw (tangent to the lane in the given point)
        """
        lane_frenet = MapUtils.get_lane_frenet_frame(lane_id)
        cpoint = lane_frenet.fpoint_to_cpoint(frenet_point)
        global_yaw = relative_yaw + lane_frenet.get_yaw(np.array([frenet_point[FP_SX]]))[0]
        return cpoint, global_yaw

    @staticmethod
    @raises(RoadNotFound, LongitudeOutOfRoad)
    def _get_lookahead_points(initial_lane_id: int, initial_lon: float, lookahead_dist: float, desired_lat: float,
                              navigation_plan: NavigationPlanMsg):
        """
        Given a longitude on specific road, return all the points along this (and next) road(s) until reaching
        a lookahead of exactly <desired_lon> meters ahead. In addition, shift all points <desired_lat_shift> laterally,
        relative to the roads right-side.
        :param initial_lane_id: the initial lane_id (the vehicle is current on)
        :param initial_lon: initial longitude along <initial_road_id>
        :param lookahead_dist: the desired distance of lookahead in [m].
        :param desired_lat: desired lateral shift of points **relative to lane center**
        :param navigation_plan: the relevant navigation plan to iterate over its road IDs.
        :return: a numpy array of points size Nx2, and the yaw of initial road longitude [rad]
        """
        map_api = MapService.get_instance()
        # find the final point's (according to desired lookahead distance) road_segment_id and longitude along this road
        lane_ids, final_lon = MapUtils._advance_on_plan(initial_lane_id, initial_lon, lookahead_dist, navigation_plan)

        # exact projection of the initial point and final point on the road
        init_pos, init_yaw = MapUtils._convert_from_lane_to_map_coordinates(initial_lane_id, np.array([initial_lon, desired_lat]))
        final_pos, _ = MapUtils._convert_from_lane_to_map_coordinates(lane_ids[-1], np.array([final_lon, desired_lat]))

        # shift points (laterally) and concatenate all points of all relevant roads
        shifted_points = np.concatenate([MapUtils._shift_lane_points_by_latitude(lid, desired_lat) for lid in lane_ids])

        # calculate accumulate longitudinal distance for all points
        longitudes = np.cumsum(np.concatenate([np.append([0], np.diff(map_api._longitudes[lid])) for lid in lane_ids]))

        # trim shifted points from both sides according to initial point and final (desired) point
        shifted_points = shifted_points[np.greater(longitudes - initial_lon, 0) &
                                        np.less(longitudes - initial_lon, lookahead_dist)]

        # Build path
        path = np.concatenate(([init_pos], shifted_points, [final_pos]))

        # Remove duplicate points (start of next road == end of current road)
        path = path[np.append(np.sum(np.diff(path, axis=0), axis=1) != 0.0, [True])]

        return path, init_yaw

    @staticmethod
    @raises(RoadNotFound, LongitudeOutOfRoad)
    def _advance_on_plan(initial_lane_id: int, initial_lon: float, lookahead_dist: float,
                         navigation_plan: NavigationPlanMsg) -> [List[int], float]:
        """
        Given a longitude on specific lane (<initial_lane_id> and <initial_lon>), advance <lookahead_dist>
        distance, return list of lane_ids in the way and final longitude w.r.t. the last lane.
        The lookahead iterates over the next roads specified in the <navigation_plan>.
        If <desired_lon> is more than the distance to end of the plan, a LongitudeOutOfRoad exception is thrown.
        :param initial_lane_id: the initial lane_id (the vehicle is current on)
        :param initial_lon: initial longitude along <initial_lane_id>
        :param lookahead_dist: the desired distance of lookahead in [m].
        :param navigation_plan: the relevant navigation plan to iterate over its road IDs.
        :return: (list of lane_ids, longitudinal distance [m] from the beginning of the last lane in lane_ids)
        """
        initial_road_segment_id = MapUtils.get_road_segment_id_from_lane_id(initial_lane_id)
        current_road_idx_in_plan = navigation_plan.get_road_index_in_plan(initial_road_segment_id)
        road_segment_ids = navigation_plan.road_ids[current_road_idx_in_plan:]

        # collect relevant lane_ids with their lengths
        downstream_lanes = MapUtils.get_downstream_lanes(initial_lane_id)
        lane_lengths = [MapUtils.get_lane_length(initial_lane_id)]
        cumulative_length = lane_lengths[0] - initial_lon
        lane_ids = [initial_lane_id]
        for road_segment_id in road_segment_ids[1:]:
            next_lane = [lid for lid in downstream_lanes if MapUtils.get_road_segment_id_from_lane_id(lid) == road_segment_id]
            if cumulative_length > lookahead_dist:
                break
            if len(next_lane) < 1:
                raise RoadNotFound("Downstream lane was not found in navigation plan")
            lane_ids.append(next_lane[0])
            current_lane_length = MapUtils.get_lane_length(next_lane[0])
            cumulative_length += current_lane_length
            lane_lengths.append(current_lane_length)
            downstream_lanes = MapUtils.get_downstream_lanes(next_lane[0])

        # distance to roads-ends
        lanes_dist_to_end = np.cumsum(np.append([lane_lengths[0] - initial_lon], lane_lengths[1:]))
        # how much of lookahead_dist is left after this road
        lanes_leftovers = np.subtract(lookahead_dist, lanes_dist_to_end)

        try:
            target_lane_idx = np.where(lanes_leftovers < 0)[0][0]
            return lane_ids[:(target_lane_idx+1)], lanes_leftovers[target_lane_idx] + lane_lengths[target_lane_idx]
        except IndexError:
            raise LongitudeOutOfRoad("The specified navigation plan is short {} meters to advance {} in longitude"
                                     .format(lanes_leftovers[-1], lookahead_dist))

    @staticmethod
    def _shift_lane_points_by_latitude(lane_id: int, lateral_shift: float) -> np.array:
        """
        Given a lane, shift its center-points laterally by <lateral_shift>.
        :param lane_id: lane id
        :param lateral_shift: [m] shift in meters
        :return: shifted points array (Nx2)
        """
        points = MapService.get_instance()._lane_points[lane_id]
        # calculate direction unit vectors
        points_direction = np.diff(points, axis=0)
        norms = np.linalg.norm(points_direction, axis=1)[np.newaxis].T
        norms[np.where(norms == 0.0)] = 1.0
        direction_unit_vec = np.divide(points_direction, norms)
        # calculate normal unit vectors
        normal_unit_vec = np.c_[-direction_unit_vec[:, 1], direction_unit_vec[:, 0]]
        normal_unit_vec = np.concatenate((normal_unit_vec, normal_unit_vec[-1, np.newaxis]))
        # shift points by lateral_shift
        shifted_points = points + normal_unit_vec * lateral_shift
        return shifted_points
