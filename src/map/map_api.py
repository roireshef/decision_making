from abc import ABCMeta
import numpy as np

from decision_making.src.exceptions import RoadNotFound, raises, LongitudeOutOfRoad
from decision_making.src.global_constants import *
from decision_making.src.map.constants import *
from decision_making.src.map.map_model import MapModel
from typing import List, Union
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.planning.utils.geometry_utils import CartesianFrame
from logging import Logger
from typing import List

import numpy as np
import six

from decision_making.src.exceptions import RoadNotFound, raises, LongitudeOutOfRoad
from decision_making.src.global_constants import *
from decision_making.src.map.map_model import MapModel
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.planning.utils.geometry_utils import CartesianFrame


@six.add_metaclass(ABCMeta)
class MapAPI:
    def __init__(self, map_model, logger):
        # type: (MapModel, Logger) -> None
        self._cached_map_model = map_model
        self.logger = logger

    def update_perceived_roads(self):
        pass

    # TODO: explain better what this does.
    # #TODO: also, consider moving ROADS_MAP_TILE_SIZE to be a member of MapModel. what is layer in coordinates?
    def find_roads_containing_point(self, layer, x, y):
        # type: (int, float, float) -> List[int]
        """
        shortcut to a cell of the map xy2road_map
        :param layer: 0 ground, 1 on bridge, 2 bridge above bridge, etc
        :param x: world coordinates in meters
        :param y: world coordinates in meters
        :return: road_ids containing the point (x, y)
        """
        cell_x = int(round(x / ROADS_MAP_TILE_SIZE))
        cell_y = int(round(y / ROADS_MAP_TILE_SIZE))
        return self._cached_map_model.get_xy2road_cell((layer, cell_x, cell_y))

    # TODO: unnecessary
    def get_road_main_details(self, road_id):
        # type: (int) -> (int, float, float, np.ndarray)
        """
        get details of a given road
        :param road_id:
        :return: lanes number, road width, road length, road's points
        """
        if road_id not in self._cached_map_model.roads_data.keys():
            return None, None, None, None
        road_details = self._get_road(road_id)
        return road_details.lanes_num, road_details.width, road_details.longitudes[-1], road_details.points

    # TODO: unnecessary. this is a two-liner.
    def get_uniform_path_lookahead(self, road_id, lat_shift, starting_lon, lon_step, steps_num, navigation_plan):
        # type: (int, float, float, float, int, NavigationPlanMsg) -> np.ndarray
        """
        Create array of uniformly distributed points along a given road, shifted laterally by by lat_shift.
        When some road finishes, it automatically continues to the next road, according to the navigation plan.
        The distance between consecutive points is lon_step.
        :param road_id: starting road_id
        :param lat_shift: lateral shift
        :param starting_lon: starting longitude
        :param lon_step: distance between consecutive points
        :param steps_num: output points number
        :param navigation_plan: the relevant navigation plan to iterate over its road IDs.
        :return: uniform points array (Nx2)
        """
        shifted = self.get_lookahead_points(road_id, starting_lon, lat_shift, lon_step * steps_num, navigation_plan)
        resampled, _ = CartesianFrame.resample_curve(shifted, lon_step)
        return resampled

    # TODO: rewrite. split into few methods? the name doesn't explain the function
    def convert_world_to_lat_lon(self, x, y, z, yaw):
        # type: (float, float, float, float) -> (int, int, float, float, float, float)
        """
        Given 3D world point, calculate:
            1. road_id,
            2. lane from left,
            3. latitude relatively to the road's left edge,
            4. longitude relatively to the road's start
            5. yaw relatively to the road
        The function uses the rendered map that for every square meter stores the road_id containing it.
        If the point is outside any road, return road_id according to the navigation plan.
        :param x:
        :param y:
        :param z:
        :param yaw:
        :return: road_id, lane, full latitude, lane_lat, longitude, yaw_in_road
        """
        # use road_id by navigation if the point is outside the roads
        road_ids = self.find_roads_containing_point(DEFAULT_MAP_LAYER, x, y)

        if len(road_ids) == 0:
            raise Exception("convert_world_to_lat_lon failed to find the road")

        # find the closest road to (x,y) among the road_ids list
        (lat_dist, sign, lon, road_yaw, road_id) = self._find_closest_road(x, y, road_ids)

        road_details = self._get_road(road_id)
        lanes_num = road_details.lanes_num
        lane_width = road_details.width / float(lanes_num)

        # calc lane number, intra-lane lat and yaw
        full_lat = lat_dist * sign + 0.5 * lanes_num * lane_width  # latitude relatively to the right road edge
        lane = float(int(full_lat / lane_width))  # from right to left
        lane = np.clip(lane, 0, lanes_num - 1)
        yaw_in_road = (yaw - road_yaw + 2 * np.pi) % (2 * np.pi)
        lane_lat = full_lat % lane_width
        return road_id, lane, full_lat, lane_lat, lon, yaw_in_road

    # TODO: rewrite as vector ops + raise exception and drop boolean
    # TODO: change to work with Nx2
    @raises(RoadNotFound)
    def get_point_relative_longitude(self, from_road_id, from_lon_in_road, to_road_id, to_lon_in_road,
                                     max_lookahead_distance, navigation_plan):
        # type: (int, float, int, float, float, NavigationPlanMsg) -> float
        """
        Find longitude distance between two points in road coordinates.
        First search forward from the point (from_road_id, from_lon_in_road) to the point (to_road_id, to_lon_in_road);
        if not found then search backward.
        :param from_road_id:
        :param from_lon_in_road: search from this point
        :param to_road_id:
        :param to_lon_in_road: search to this point
        :param max_lookahead_distance: max search distance
        :return: longitude distance between the given two points, or raises an exception if connection is not found
        """
        if to_road_id == from_road_id:  # simple case
            total_lon_distance = to_lon_in_road - from_lon_in_road
            return total_lon_distance

        road_id = from_road_id
        found_connection = False
        total_lon_distance = max_lookahead_distance

        # first search forward (direction=1); if not found then search backward (direction=-1)
        for direction in [1, -1]:

            # 1. First road segment
            if direction > 0:  # forward
                total_lon_distance = self._get_road(from_road_id).longitudes[-1] - from_lon_in_road
            else:  # backward
                total_lon_distance = from_lon_in_road

            # 2. Middle road segments
            road_id = navigation_plan.get_next_road(road_id)
            while road_id is not None and road_id != to_road_id and total_lon_distance < max_lookahead_distance:
                road_length = self._get_road(road_id).longitudes[-1]
                total_lon_distance += road_length
                road_id = navigation_plan.get_next_road(road_id)

            # 3. Add length of last road segment
            if road_id == to_road_id:
                if direction > 0:  # forward
                    total_lon_distance += to_lon_in_road
                else:  # backward
                    total_lon_distance += self._get_road(to_road_id).longitudes[-1] - to_lon_in_road
                found_connection = True
                break  # stop the search when the connection is found

        if not found_connection:
            raise RoadNotFound("The connection within max_lookahead_distance {} between source road_id \
                               {} and destination road_id {} was not found".format(max_lookahead_distance,
                                                                                   from_road_id, to_road_id))

        return total_lon_distance

    @raises(RoadNotFound)
    def get_center_lanes_latitudes(self, road_id):
        # type: (int) -> np.array
        """
        Get list of latitudes of all centers of lanes in the road
        :param road_id: Road ID to iterate over its lanes
        :return: list of latitudes of all centers of lanes in the road relative to the right side of the road
        """
        road_details = self._get_road(road_id)
        lanes_num = road_details.lanes_num
        road_width = road_details.width
        lane_width = float(road_width) / lanes_num
        center_lanes = lane_width / 2 + np.array(range(lanes_num)) * lane_width
        return center_lanes

    @raises(RoadNotFound, LongitudeOutOfRoad)
    def get_lookahead_points(self, initial_road_id, initial_lon, desired_lon, desired_lat_shift, navigation_plan):
        # type: (int, float, float, float, NavigationPlanMsg) -> np.ndarray
        """
        Given a longitude on specific road, return all the points along this (and next) road(s) until reaching
        a lookahead of exactly <desired_lon> meters ahead. In addition, shift all points <desired_lat_shift> laterally,
        relative to the roads right-side.
        :param initial_road_id: the initial road_id (the vehicle is current on)
        :param initial_lon: initial longitude along <initial_road_id>
        :param desired_lon: the desired distance of lookahead in [m].
        :param desired_lat_shift: desired lateral shift **relative to road's right-side**
        :param navigation_plan: the relevant navigation plan to iterate over its road IDs.
        :return:
        """
        # find the final point's (according to desired lon. lookahead amount) road_id and longitude along this road
        final_road_id, final_road_lon = self.advance_on_plan(initial_road_id, initial_lon, desired_lon, navigation_plan)
        init_road_idx = navigation_plan.get_road_index_in_plan(initial_road_id)
        final_road_idx = navigation_plan.get_road_index_in_plan(final_road_id)
        relevant_road_ids = navigation_plan.road_ids[init_road_idx:(final_road_idx+1)]

        # exact projection of the initial point and final point on the road
        init_point = self._convert_road_to_global_coordinates(initial_road_id, initial_lon, desired_lat_shift)[0][:2]
        final_point = self._convert_road_to_global_coordinates(final_road_id, final_road_lon, desired_lat_shift)[0][:2]

        # shift points (laterally) and concatenate all points of all relevant roads
        shifted_points = np.concatenate([self._shift_road_points_from_rightside(rid, desired_lat_shift)
                                         for rid in relevant_road_ids])

        # calculate accumulate longitudinal distance for all points
        longitudes = np.cumsum(np.concatenate([np.append([0], np.diff(self._get_road(rid).longitudes))
                                               for rid in relevant_road_ids]))

        # trim shifted points from both sides according to initial point and final (desired) point
        shifted_points = shifted_points[np.greater(longitudes - initial_lon, 0) &
                                        np.less(longitudes - initial_lon, desired_lon)]

        path = np.concatenate(([init_point], shifted_points, [final_point]))
        return path[np.append(np.sum(np.diff(path, axis=0), axis=1) != 0.0, [True])]

    @raises(RoadNotFound, LongitudeOutOfRoad)
    def advance_on_plan(self, initial_road_id, initial_lon, desired_lon, navigation_plan):
        # type: (int, float, float, NavigationPlanMsg) -> (int, float)
        """
        Given a longitude on specific road (<initial_road_id> and <initial_lon>), advance (lookahead) <desired_lon>
        distance. The lookahead iterates over the next roads specified in the <navigation_plan> and returns: (the final
        road id, the longitude along this road). If <desired_lon> is more than the distance to end of the plan, a
        LongitudeOutOfRoad exception is thrown.
        :param initial_road_id: the initial road_id (the vehicle is current on)
        :param initial_lon: initial longitude along <initial_road_id>
        :param desired_lon: the deisred distance of lookahead in [m].
        :param navigation_plan: the relevant navigation plan to iterate over its road IDs.
        :return: (road_id, longitude along <road_id>)
        """
        current_road_idx_in_plan = navigation_plan.get_road_index_in_plan(initial_road_id)
        roads_ids = navigation_plan.road_ids[current_road_idx_in_plan:]
        roads_len = [self._cached_map_model.get_road_data(rid).longitudes[-1] for rid in roads_ids]

        roads_dist_to_end = np.cumsum(np.append([roads_len[0] - initial_lon], roads_len[1:]))  # dist to roads-ends
        roads_leftovers = np.subtract(desired_lon, roads_dist_to_end)  # how much of step_lon is left after this road

        try:
            target_road_idx = np.where(roads_leftovers < 0)[0][0]
            return roads_ids[target_road_idx], roads_leftovers[target_road_idx] + roads_len[target_road_idx]
        except IndexError:
            raise LongitudeOutOfRoad("The specified navigation plan is short %f meters to advance %f in longitude",
                                     roads_leftovers[-1], desired_lon)

    @raises(RoadNotFound)
    def advance_to_end_of_plan(self, initial_road_id, initial_lon, navigation_plan):
        # type: (int, float, NavigationPlanMsg) -> (int, float, float)
        """
        Given a longitude on specific road (<initial_road_id> and <initial_lon>), advance (lookahead) to the final point
        in the navigation plan.
        :param initial_road_id: the initial road_id (the vehicle is current on)
        :param initial_lon: initial longitude along <initial_road_id>
        :param navigation_plan: the relevant navigation plan to iterate over its road IDs.
        :return: (the last road id, its length, accumulated distance its end point)
        """
        initial_road_idx_in_plan = navigation_plan.get_road_index_in_plan(initial_road_id)
        roads_ids = navigation_plan.road_ids[initial_road_idx_in_plan:]
        roads_len = [self._cached_map_model.get_road_data(rid).longitudes[-1] for rid in roads_ids]
        roads_dist_to_end = np.cumsum(np.append([roads_len[0] - initial_lon], roads_len[1:]))  # dist to roads-ends
        return roads_ids[-1], roads_len[-1], roads_dist_to_end[-1]

    @raises(RoadNotFound, LongitudeOutOfRoad)
    def get_global_point_on_road_rightside(self, road_id, lon):
        road_width = self._cached_map_model.get_road_data(road_id).width
        return self._convert_road_to_global_coordinates(road_id, lon, -road_width / 2)

    '''###################'''
    ''' PRIVATE FUNCTIONS '''
    '''###################'''

    @raises(RoadNotFound)
    def _get_road(self, road_id):
        return self._cached_map_model.get_road_data(road_id)

    @staticmethod
    def _shift_road_points(points, lateral_shift):
        # type: (np.ndarray, float) -> np.ndarray
        """
        Given points list along a road, shift them laterally by lat_shift meters
        :param points (Nx2): points list along a given road
        :param lateral_shift: shift in meters
        :return: shifted points array (Nx2)
        """
        points_direction = np.diff(points, axis=0)
        direction_unit_vec = MapAPI._normalize_matrix_rows(points_direction)
        normal_unit_vec = np.c_[-direction_unit_vec[:, 1], direction_unit_vec[:, 0]]
        normal_unit_vec = np.concatenate((normal_unit_vec, normal_unit_vec[-1, np.newaxis]))
        shifted_points = points + normal_unit_vec * lateral_shift
        return shifted_points

    @raises(RoadNotFound)
    def _shift_road_points_from_rightside(self, road_id, latitude_shift):
        """
        Returns Road.points shifted by <latitude_shift> relative to road's right-side
        :param road_id: road ID to get the points of.
        :param latitude_shift: desired shift relative to road's right-side
        :return:
        """
        road = self._get_road(road_id)
        return self._shift_road_points_in_latitude(road.points, latitude_shift - road.width / 2)

    @raises(RoadNotFound)
    def _find_closest_road(self, x, y, road_ids):
        # type: (float, float, List[int]) -> (float, float, int)
        """
        Returns the closest road_id of the road which is closest to a point in the world (x, y).
        If the point is on the road (in the sense of longitude), closest_lat is also the distance between the
        point and the road. Otherwise closest_lat is the distance but not latitude, because in this case latitude
        is meaningless.
        :param x: x coordinate on map (given in [m])
        :param y: y coordinate on map (given in [m])
        :param road_ids: list of road IDs to try to project the point on
        :return: (lat [m], lon [m], road_id) from the closest road
        """
        closest_lat = closest_lon = np.inf
        closest_id = None
        for road_id in road_ids:
            lat, lon = self._convert_global_to_road_coordinates(x, y, road_id)
            if np.math.fabs(lat) < np.math.fabs(closest_lat):
                closest_lat, closest_lon, closest_id = lat, lon, road_id
        return closest_lat, closest_lon, closest_id

    @raises(RoadNotFound, LongitudeOutOfRoad)
    def _convert_road_to_global_coordinates(self, road_id, lon, lat):
        # type: (int, float, float) -> (np.array, float)
        """
        Given road ID, longitude and latitude along it (relative to its center points), find the matching point in
        global (cartesian) coordinate frame.
        :param road_id: road ID as in the map model
        :param lon: longitude from the beginning of the current road
        :param lat: latitude relative to road's center points
        :return: numpy array of 3D point [x, y, z] in global coordinate frame, yaw [rad] in global coordinate frame
        """
        road = self._cached_map_model.get_road_data(road_id)
        points_with_yaw = CartesianFrame.add_yaw(road.points)

        if road.longitudes[0] <= lon <= road.longitudes[-1]:
            point_ind = np.argmin(np.abs(road.longitudes - lon))  # find index closest to target lon
            distance_in_lon_from_closest_point = lon - road.longitudes[point_ind]
            road_point = points_with_yaw[point_ind]

            # Move lat from the rightmost edge of road
            # Also, fix move along the lon axis by 'distance_in_lon_from_closest_point',
            # in order to fix the difference caused by the map quantization.
            lon_lat_shift = np.array([distance_in_lon_from_closest_point, lat - road.width / 2, 1])
            shifted_point = np.dot(CartesianFrame.homo_matrix_2d(road_point[2], road_point[:2]), lon_lat_shift)

            position_in_world = np.append(shifted_point[:2], [DEFAULT_OBJECT_Z_VALUE])
            orientation_in_world = road_point[2]
            return position_in_world, orientation_in_world
        else:
            raise LongitudeOutOfRoad("longitude {} is out of road's longitudes range [{}, {}]"
                                     .format(lon, road.longitudes[0], road.longitudes[-1]))

    @raises(RoadNotFound)
    def _convert_global_to_road_coordinates(self, x, y, road_id):
        # type: (float, float, int) -> (float, float)
        """
        Convert point in world coordinates (x, y) to (lat, lon) of road with given road_id
        If the point is on the road (in the sense of longitude), then lat is also the distance between the point
        and the road. Otherwise lat is the distance but not latitude, because in this case latitude is meaningless.
        :param x: the point's world x coordinate in meters
        :param y: the point's world y coordinate in meters
        :param road_id: road ID as in the map model
        :return: signed lat (relatively to the road center), lon (from road start)
        """
        p = np.array([x, y])
        road = self._cached_map_model.get_road_data(road_id)
        longitudes = road.longitudes

        # find the closest point of the road to (x,y)
        points = road.points[:, 0:2]
        distance_to_road_points = np.linalg.norm(np.array(points) - p, axis=0)
        closest_point_ind = np.argmin(distance_to_road_points)

        # the relevant road segments will be the one before this point, and the one after it, so for both segments:
        # compute [sign, latitude, longitude, segment_start_point_index]
        closest_point_ind_pairs = [[closest_point_ind - 1, closest_point_ind], [closest_point_ind, closest_point_ind + 1]]
        segments = [np.append(CartesianFrame.calc_point_segment_dist(p, points[pair[0]], points[pair[1]]), pair[0])
                    for pair in closest_point_ind_pairs if pair[0] >= 0 and pair[1] < len(points)]
        segments = np.array(segments)

        # find closest segment by min distance (latitude)
        closest_segment = segments[np.argmin(segments[:, 1], axis=0)]
        sign, lat, lon, start_ind = closest_segment[0], closest_segment[1], closest_segment[2], int(closest_segment[3])

        # lat, lon
        return road.width / 2 + lat * sign, lon + longitudes[start_ind]

    @staticmethod
    def _normalize_matrix_rows(mat):
        # type: (np.array) -> np.array
        """
        normalize vector, prevent division by zero
        :param mat: 2D numpy array
        :return: normalized vector (numpy array)
        """
        norms = np.linalg.norm(mat, axis=1)[np.newaxis].T
        norms[np.where(norms == 0.0)] = 1.0
        return np.divide(mat, norms)
