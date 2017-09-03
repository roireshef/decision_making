from logging import Logger
from typing import List, Union

import numpy as np

from decision_making.src.exceptions import RoadNotFound, raises, LongitudeOutOfRoad
from decision_making.src.global_constants import MAP_NAME_FOR_LOGGING
from decision_making.src.map.map_model import MapModel
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.planning.utils.geometry_utils import CartesianFrame
from rte.python.logger.AV_logger import AV_Logger


class MapAPI:
    def __init__(self, map_model, logger):
        # type: (MapModel, Logger) -> None
        self._cached_map_model = map_model
        self.logger = logger

    def find_roads_containing_point(self, layer, world_x, world_y):
        # type: (int, float, float) -> List[int]
        """
        shortcut to a cell of the map xy2road_map
        :param layer: 0 ground, 1 on bridge, 2 bridge above bridge, etc
        :param world_x: world coordinates in meters
        :param world_y: world coordinates in meters
        :return: road_ids containing the point (world_x, world_y)
        """
        pass

    def get_center_lanes_latitudes(self, road_id):
        # type: (int) -> np.array
        """
        get list of latitudes of all centers of lanes in the road
        :param road_id:
        :return: list of latitudes of all centers of lanes in the road relative to the right side of the road
        """
        pass

    def get_road_main_details(self, road_id):
        # type: (int) -> (int, float, float, np.ndarray)
        """
        get details of a given road
        :param road_id:
        :return: lanes number, road width, road length, road's points
        """
        pass

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
        pass

    def get_point_relative_longitude(self, from_road_id, from_lon_in_road, to_road_id, to_lon_in_road,
                                     max_lookahead_distance, navigation_plan):
        # type: (int, float, int, float, float, NavigationPlanMsg) -> (float, bool)
        """
        Find longitude distance between two points in road coordinates.
        First search forward from the point (from_road_id, from_lon_in_road) to the point (to_road_id, to_lon_in_road);
        if not found then search backward.
        :param from_road_id:
        :param from_lon_in_road: search from this point
        :param to_road_id:
        :param to_lon_in_road: search to this point
        :param max_lookahead_distance: max search distance
        :return: longitude distance between the given two points, boolean "found connection"
        """
        pass

    def get_path_lookahead(self, road_id, lon, lat, max_lookahead_distance, navigation_plan, direction=1):
        # type: (int, float, float, float, NavigationPlanMsg, int) -> Union[np.ndarray, None]
        """
        Get path with lookahead distance (starting from certain road, and continuing to the next ones if lookahead distance > road length)
            lat is measured in meters
        The function returns original roads points shifted by lat, rather than uniformly distanced points
        :param road_id: starting road_id
        :param lon: starting lon
        :param lat: lateral shift in meters
        :param max_lookahead_distance:
        :param direction: forward (1) or backward (-1)
        :return: points array
        """
        pass

    def get_uniform_path_lookahead(self, road_id, lat, starting_lon, lon_step, steps_num, navigation_plan):
        # type: (int, float, float, float, int, NavigationPlanMsg) -> np.ndarray
        """
        Create array of uniformly distanced points along the given road, shifted by lat.
        When some road finishes, it automatically continues to the next road, according to the navigation plan.
        The distance between consecutive points is lon_step.
        :param road_id: starting road_id
        :param lat: lateral shift
        :param starting_lon: starting longitude
        :param lon_step: distance between consecutive points
        :param steps_num: output points number
        :return: uniform points array (2xN)
        """
        pass

    def update_perceived_roads(self):
        pass

    def _find_closest_road(self, x, y, road_ids):
        # type: (float, float, List[int]) -> (float, float, int)
        """
        Returns the closest road_id of the road which is closest to a point in the world (x, y)
        :param x: x coordinate on map (given in [m])
        :param y: y coordinate on map (given in [m])
        :param road_ids: list of road_id
        :return: (lat [m], lon [m], road_id) from the closest road
        """
        closest_lat = closest_lon = np.inf
        closest_id = None
        for road_id in road_ids:
            lat, lon = self._convert_global_to_road_coordinates(x, y, road_id)
            if np.math.fabs(lat) < np.math.fabs(closest_lat):
                closest_lat, closest_lon, closest_id = lat, lon, road_id
        return closest_lat, closest_lon, closest_id

    @staticmethod
    def _shift_road_points_in_latitude(points, lat_shift):
        # type: (np.ndarray, float) -> np.ndarray
        """
        Given points list along a road, shift them laterally by lat_shift meters
        :param points (Nx2): points list along a given road
        :param lat_shift: shift in meters
        :return: shifted points array (Nx2)
        """
        points_direction = np.diff(points, axis=0)
        norms = np.linalg.norm(points_direction, axis=1)[np.newaxis].T
        if not np.all(np.greater(norms, 0.0)):
            # TODO: find a better way
            MapAPI.logger.warning('Identical consecutive points in path. Norm of diff is Zero')
        direction_unit_vec = np.divide(points_direction, norms)
        normal_unit_vec = np.c_[-direction_unit_vec[:, 1], direction_unit_vec[:, 0]]
        normal_unit_vec = np.concatenate((normal_unit_vec, normal_unit_vec[-1, np.newaxis]))
        shifted_points = points + normal_unit_vec * lat_shift
        return shifted_points

    @raises(RoadNotFound, LongitudeOutOfRoad)
    def advance_on_plan(self, initial_road_id, initial_lon, desired_lon, navigation_plan):
        # type: (int, float, float, NavigationPlanMsg) -> (int, float)
        """
        Given a longitude on specific road (<initial_road_id> and <initial_lon>), advance (lookahead) <desired_lon>
        distance. The lookahead iterates over the next roads specified in the <navigation_plan> and returns: (the final
        road id, the longitude along this road). If <desired_lon> is more than the distnace to end of the plan, a
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
    def _convert_road_to_global_coordinates(self, road_id, lon, lat):
        # type: (int, float, float) -> (np.array, float)
        """
        get the global coordinate that corresponds to a given road ID, longitude and latitude (relative to its
        center points) along it.
        :return:
            numpy array of 3D point [x, y, z] in global coordinate frame;
            yaw [rad] in global coordinate frame
        """
        road = self._cached_map_model.get_road_data(road_id)
        points_with_yaw = CartesianFrame.add_yaw(road.points)

        if road.longitudes[0] <= lon <= road.longitudes[-1]:
            pnt_ind = np.argmin(np.abs(road.longitudes - lon)) # find index closest to target lon
            distance_in_lon_from_closest_point = lon - road.longitudes[pnt_ind]
            road_point = points_with_yaw[pnt_ind]

            # Move lat from the rightmost edge of road
            # Also, fix move along the lon axis by 'distance_in_lon_from_closest_point',
            # in order to fix the difference caused by the map quantization.
            lon_lat_shift = np.array([distance_in_lon_from_closest_point, lat - road.width/2, 1])
            shifted_point = np.dot(CartesianFrame.homo_matrix_2d(road_point[2], road_point[:2]), lon_lat_shift)

            #TODO: currently we assume altitude z = 0
            position_in_world = np.append(shifted_point[:2], [0.])
            orientation_in_world = road_point[2]
            return position_in_world, orientation_in_world
        else:
            raise LongitudeOutOfRoad("longitude %f is out of road's longitudes range [%f, %f]",
                                     lon, road.longitudes[0], road.longitudes[-1])

    @raises(RoadNotFound, LongitudeOutOfRoad)
    def get_global_point_on_road_rightside(self, road_id, lon):
        road_width = self._cached_map_model.get_road_data(road_id).width
        return self._convert_road_to_global_coordinates(road_id, lon, -road_width/2)

    @raises(RoadNotFound)
    def _convert_global_to_road_coordinates(self, x, y, road_id):
        # type: (float, float, int) -> (float, float)
        """
        Convert point in world coordinates (x, y) to (lat, lon) of road with given road_id
        :param x: the point's world x coordinate in meters
        :param y: the point's world y coordinate in meters
        :param road_id:
        :return: signed lat (relatively to the road center), lon (from road start)
        """
        p = np.array([x, y])
        road_details = self._cached_map_model.get_road_data(road_id)
        longitudes = road_details.longitudes

        # find the closest point of the road to (x,y)
        points = road_details.points[:, 0:2]
        distance_to_road_points = np.linalg.norm(np.array(points) - p, axis=0)
        closest_pnt_ind = np.argmin(distance_to_road_points)
        
        # the relevant road segments will be the one before this point, and the one after it, so for both segments:
        # compute [sign, latitude, longitude, segment_start_point_index]
        closest_pnt_ind_pairs = [[closest_pnt_ind - 1, closest_pnt_ind], [closest_pnt_ind, closest_pnt_ind + 1]]
        segments = np.array([np.append(CartesianFrame.calc_point_segment_dist(p, points[idxs[0]], points[idxs[1]]), idxs[0])
                             for idxs in closest_pnt_ind_pairs
                             if idxs[0] >= 0 and idxs[1] < len(points)])

        # find closest segment by min latitude
        closest_segment = segments[np.argmin(segments[:, 1], axis=0)]
        sign, lat, lon, start_ind = closest_segment[0], closest_segment[1], closest_segment[2], int(closest_segment[3])

        # lat, lon
        return road_details.width / 2 + lat * sign, lon + longitudes[start_ind]

    @staticmethod
    def _normalize_vec(vec):
        # type: (np.array) -> np.array
        """
        normalize vector, prevent division by zero
        :param vec: numpy array
        :return: normalized vector (numpy array)
        """
        vec_norm = np.linalg.norm(vec)
        if vec_norm != 0:
            return vec / vec_norm
        else:
            return vec
