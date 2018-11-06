from typing import List

import numpy as np

from decision_making.src.global_constants import TRAJECTORY_ARCLEN_RESOLUTION
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.planning.behavioral.data_objects import RelativeLane
from decision_making.src.planning.types import FP_DX, FP_SX, C_X, C_Y
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from mapping.src.exceptions import raises, LongitudeOutOfRoad, RoadNotFound
from mapping.src.service.map_service import MapService
from mapping.src.transformations.geometry_utils import CartesianFrame


class MapUtils:

    @staticmethod
    def get_road_segment_by_lane(lane_id: int) -> int:
        """
        get road_id containing the lane
        :param lane_id:
        :return: road_id
        """
        return MapService.get_instance()._lane_address[lane_id][0]

    @staticmethod
    def get_lane_index(lane_id: int) -> int:
        """
        get lane index of the lane on the road (the rightest lane's index is 0)
        :param lane_id:
        :return: lane index
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
    def get_adjacent_lane(lane_id: int, relative_lane: RelativeLane) -> int:
        """
        get adjacent (right/left) lane to the given lane segment (if exists)
        :param lane_id:
        :param relative_lane: either right or left
        :return: adjacent lane id or None if it does not exist
        """
        map_api = MapService.get_instance()
        road_id, lane_idx = map_api._lane_address[lane_id]
        adjacent_idx = lane_idx + relative_lane.value
        return map_api._lane_by_address[(road_id, adjacent_idx)] \
            if (road_id, adjacent_idx) in map_api._lane_by_address else None

    @staticmethod
    def get_closest_lane(x: float, y: float, road_segment_id: int=None) -> int:
        """
        given cartesian coordinates, find the closest lane to the point
        :param x: X cartesian coordinate
        :param y: Y cartesian coordinate
        :param road_segment_id: optional argument for road_segment_id closest to the given point
        :return: closest lane segment id
        """
        map_api = MapService.get_instance()
        if road_segment_id is None:
            # find the closest road segment
            relevant_road_ids = map_api._find_roads_containing_point(x, y)
            closest_road_id = map_api._find_closest_road(x, y, relevant_road_ids)
        else:
            closest_road_id = road_segment_id

        # find the closest lane segment, given the closest road segment
        num_lanes = map_api.get_road(closest_road_id).lanes_num
        # convert the given cpoint to fpoints w.r.t. to each lane's frenet frame
        fpoints = {}
        for lane_idx in range(num_lanes):
            lane_id = map_api._lane_by_address[(closest_road_id, lane_idx)]
            fpoints[lane_id] = map_api._lane_frenet[lane_id].cpoint_to_fpoint(np.array([x, y]))
        # find frenet points with minimal absolute latitude
        return min(fpoints.items(), key=lambda p: abs(p[1][FP_DX]))[0]

    @staticmethod
    def get_dist_from_lane_borders(lane_id: int, s: float) -> (float, float):
        """
        get distance from the lane center to the lane borders at given longitude from the lane's origin
        :param lane_id:
        :param s: longitude of the lane center point (w.r.t. the lane Frenet frame)
        :return: distance from the right lane border, distance from the left lane border
        """
        # this implementation assumes constant lane width (ignores the argument s)
        lane_width = MapService.get_instance().get_road(MapUtils.get_road_segment_by_lane(lane_id)).lane_width
        return lane_width/2, lane_width/2

    @staticmethod
    def get_dist_from_road_borders(lane_id: int, s: float) -> (float, float):
        """
        get distance from the lane center to the road borders at given longitude from the lane's origin
        :param lane_id:
        :param s: longitude of the lane center point (w.r.t. the lane Frenet frame)
        :return: distance from the right road border, distance from the left road border
        """
        # this implementation assumes constant lane width (ignores the argument s), the same width of all road's lanes
        map_api = MapService.get_instance()
        road_segment_id = MapUtils.get_road_segment_by_lane(lane_id)
        lane_width = map_api.get_road(road_segment_id).lane_width
        num_lanes = map_api.get_road(road_segment_id).lanes_num
        lane_idx = MapUtils.get_lane_index(lane_id)
        return (lane_idx + 0.5)*lane_width, (num_lanes - lane_idx - 0.5)*lane_width

    @staticmethod
    def get_lane_width(lane_id: int, s: float) -> float:
        """
        get lane width at given longitude from the lane's origin
        :param lane_id:
        :param s: longitude of the lane center point (w.r.t. the lane Frenet frame)
        :return: lane width
        """
        dist_from_right, dist_from_left = MapUtils.get_dist_from_lane_borders(lane_id, s)
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
        # in current implementation: if starting_lon < 0, extract Frenet frame with only positive longitudes
        if starting_lon < 0:
            lookahead_dist += starting_lon
            starting_lon = 0

        shifted, _ = MapUtils._get_lookahead_points(lane_id, starting_lon, lookahead_dist, desired_lat=0,
                                                    navigation_plan=navigation_plan)
        # TODO: change to precise resampling
        _, resampled, _ = CartesianFrame.resample_curve(curve=shifted, step_size=TRAJECTORY_ARCLEN_RESOLUTION)

        center_lane_reference_route = FrenetSerret2DFrame.fit(resampled)
        return center_lane_reference_route

    @staticmethod
    def get_upstream_lanes(lane_id: int) -> List[int]:
        """
        get upstream lanes of the given lane
        :param lane_id:
        :return: list of upstream lanes
        """
        pass

    @staticmethod
    def get_downstream_lanes(lane_id: int) -> List[int]:
        """
        get downstream lanes of the given lane
        :param lane_id:
        :return: list of downstream lanes
        """
        pass

    @staticmethod
    def is_main_lane(lane_id: int) -> bool:
        """
        check if there is another lane with higher priority, having the same downstream lane
        :param lane_id:
        :return: True if there is no another lane with higher priority, having the same downstream lane
        """
        pass

    @staticmethod
    def get_lanes_by_road_segment(road_segment_id: int) -> List[int]:
        """
        Get sorted list of lanes for given road segment. The output lanes are ordered by the lanes' index,
        i.e. from the rightest lane to the most left.
        :param road_segment_id:
        :return: sorted list of lane segments' IDs
        """
        map_api = MapService.get_instance()
        # find the closest lane segment, given the closest road segment
        num_lanes = map_api.get_road(road_segment_id).lanes_num
        lanes_list = []
        for lane_idx in range(num_lanes):
            lane_id = map_api._lane_by_address[(road_segment_id, lane_idx)]
            lanes_list.append(lane_id)
        return lanes_list

    @staticmethod
    def convert_from_lane_to_map_coordinates(lane_id: int, lon_on_lane: float, lat_on_lane: float,
                                             relative_yaw: float=0) -> [float, float, float]:
        """
        convert a point from lane coordinates to map (global) coordinates
        :param lane_id:
        :param lon_on_lane: longitude w.r.t. the lane
        :param lat_on_lane: latitude w.r.t. the lane
        :param relative_yaw: intra-lane yaw (optional)
        :return: map coordinates: x, y, yaw (tangent to the lane in the given point)
        """
        lane_frenet = MapUtils.get_lane_frenet_frame(lane_id)
        cpoint = lane_frenet.fpoint_to_cpoint(np.array([lon_on_lane, lat_on_lane]))
        global_yaw = relative_yaw + lane_frenet.get_yaw(np.array([lon_on_lane]))[0]
        return cpoint[C_X], cpoint[C_Y], global_yaw

    @staticmethod
    def convert_from_map_to_lane_coordinates(x: float, y: float, global_yaw: float) -> [int, float, float, float, bool]:
        """
        convert a point from map (global) coordinates to lane coordinates, by searching for the nearest lane and
        projecting it onto this lane
        :param x: X map coordinate
        :param y: Y map coordinate
        :param global_yaw: yaw in map coordinates
        :return: lane_id, longitude on the lane, latitude on the lane, intra_lane_yaw (rad),
                    is object within the road latitudes
        """
        lane_id = MapUtils.get_closest_lane(x, y)
        lane_frenet = MapUtils.get_lane_frenet_frame(lane_id)
        fpoint = lane_frenet.cpoint_to_fpoint(np.array([x, y]))  # frenet coordinates w.r.t. the lane
        relative_yaw = global_yaw - lane_frenet.get_yaw(fpoint[FP_SX])  # yaw w.r.t. the lane
        # check if the point is on road
        dist_to_right_border, dist_to_left_border = MapUtils.get_dist_from_road_borders(lane_id, fpoint[FP_SX])
        is_on_road = bool(-dist_to_right_border <= fpoint[FP_DX] <= dist_to_left_border)

        return lane_id, fpoint[FP_SX], fpoint[FP_DX], relative_yaw, is_on_road

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
        # find the final point's (according to desired lookahead distance) road_id and longitude along this road
        lane_ids, final_lon = MapUtils._advance_on_plan(initial_lane_id, initial_lon, lookahead_dist, navigation_plan)

        # exact projection of the initial point and final point on the road
        init_x, init_y, init_yaw = MapUtils.convert_from_lane_to_map_coordinates(initial_lane_id, initial_lon, desired_lat)
        init_pos = np.array([init_x, init_y])
        final_x, final_y, _ = MapUtils.convert_from_lane_to_map_coordinates(lane_ids[-1], final_lon, desired_lat)
        final_pos = np.array([final_x, final_y])

        # shift points (laterally) and concatenate all points of all relevant roads
        shifted_points = np.concatenate([MapUtils._shift_lane_points_to_latitude(lid, desired_lat) for lid in lane_ids])

        # calculate accumulate longitudinal distance for all points
        longitudes = np.cumsum(np.concatenate([np.append([0], np.diff(map_api._longitudes[lid])) for lid in lane_ids]))

        # trim shifted points from both sides according to initial point and final (desired) point
        shifted_points = shifted_points[np.greater(longitudes - initial_lon, 0) &
                                        np.less(longitudes - initial_lon, lookahead_dist)]

        # Build path
        path = np.concatenate(([init_pos], shifted_points, [final_pos]))

        # Remove duplicate points (start of next road == end of last road)
        path = path[np.append(np.sum(np.diff(path, axis=0), axis=1) != 0.0, [True])]

        return path, init_yaw

    @staticmethod
    @raises(RoadNotFound, LongitudeOutOfRoad)
    def _advance_on_plan(initial_lane_id: int, initial_lon: float, lookahead_dist: float,
                         navigation_plan: NavigationPlanMsg) -> [np.array, float]:
        """
        Given a longitude on specific road (<initial_road_id> and <initial_lon>), advance (lookahead) <desired_lon>
        distance. The lookahead iterates over the next roads specified in the <navigation_plan> and returns: (the final
        road id, the longitude along this road). If <desired_lon> is more than the distance to end of the plan, a
        LongitudeOutOfRoad exception is thrown.
        :param initial_lane_id: the initial lane_id (the vehicle is current on)
        :param initial_lon: initial longitude along <initial_lane_id>
        :param lookahead_dist: the desired distance of lookahead in [m].
        :param navigation_plan: the relevant navigation plan to iterate over its road IDs.
        :return: (road_id, longitudinal distance [m] from the beginning of <road_id>)
        """
        initial_road_id = MapUtils.get_road_segment_by_lane(initial_lane_id)
        current_road_idx_in_plan = navigation_plan.get_road_index_in_plan(initial_road_id)
        roads_ids = navigation_plan.road_ids[current_road_idx_in_plan:]

        # collect relevant lane_ids with their lengths
        downstream_lanes = MapUtils.get_downstream_lanes(initial_lane_id)
        lane_lengths = [MapUtils.get_lane_length(initial_lane_id)]
        lane_ids = [initial_lane_id]
        for road_id in roads_ids[1:]:
            next_lane = [lid for lid in downstream_lanes if MapUtils.get_road_segment_by_lane(lid) == road_id]
            if len(next_lane) < 1:
                raise RoadNotFound("Downstream lane was not found in navigation plan")
            lane_ids.append(next_lane[0])
            lane_lengths.append(MapUtils.get_lane_length(next_lane[0]))
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
    def _shift_lane_points_to_latitude(lane_id: int, lateral_shift: float) -> np.array:
        """
        Given points list along a lane, shift them laterally by lat_shift [m]
        :param lane_id: lane id
        :param lateral_shift: shift in meters
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
