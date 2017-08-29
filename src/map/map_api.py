import numpy as np

from decision_making.src.global_constants import MAP_NAME_FOR_LOGGING
from decision_making.src.map.map_model import MapModel
from typing import List, Union
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.planning.utils.geometry_utils import CartesianFrame
from logging import Logger

from decision_making.src.map.constants import LARGE_NUM
from rte.python.logger.AV_logger import AV_Logger


class MapAPI:

    logger = AV_Logger.get_logger(MAP_NAME_FOR_LOGGING)

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
        :return: (lat [m], lon [m], road_id) of the closest road
        """
        # find the closest road to (x,y)
        closest_lat = LARGE_NUM
        closest_id = closest_lon = None
        for road_id in road_ids:
            lat, lon = self._convert_world_to_lat_lon_for_given_road(x, y, road_id)
            if lat < closest_lat:
                (closest_lat, closest_lon, closest_id) = (lat, lon, road_id)
        return closest_lat, closest_lon, closest_id

    @staticmethod
    def _shift_road_vector_in_lat(points, lat_shift):
        # type: (np.ndarray, float) -> np.ndarray
        """
        Given points list along a road, shift them laterally by lat_shift meters
        :param points (2xN): points list along a given road
        :param lat_shift: shift in meters
        :return: shifted points array (2xN)
        """
        points = np.array(points)
        points_direction = np.diff(points, axis=1)
        points_norm = np.linalg.norm(points_direction, axis=0)
        if len(np.where(points_norm == 0.0)[0]) > 0:
            MapAPI.logger.warning('Identical consecutive points in path. Norm of diff is Zero')
        normalized_vec_x = np.divide(points_direction[0, :], points_norm)
        normalized_vec_y = np.divide(points_direction[1, :], points_norm)
        lat_vec = np.vstack((-normalized_vec_y, normalized_vec_x))
        lat_vec = np.concatenate((lat_vec, lat_vec[:, -1].reshape([2, 1])), axis=1)
        shifted_points = points + lat_vec * lat_shift
        return shifted_points

    def _advance_road_coordinates_in_lon(self, road_id, start_lon, lon_step, navigation_plan):
        # type: (int, float, float, NavigationPlanMsg, int) -> (int, float, float)
        """
        Get the road matching to a given longitude of a given road.
        If the given longitude exceeds the current road length, then calculate the point in the next road.
        The next road is picked from the navigation plan.
        :param road_id: current road_id
        :param start_lon: the point's longitude relatively to the start of the road_id
        :param start_lon: the step in [m] in longitude
        :param navigation_plan: of type NavigationPlan, includes list of road_ids
        :return: 1. new road_id (maybe the same one); 2. its lon; 3. residual lon from road_lon
            1. the resulted road_id may differ from the input road_id because the target point may belong to another road.
            2. for the same reason the resulted road_lon may differ from the input road_lon.
            3. If couldn't advance to road_lon (due to end of road / navigation plan), then the residual lon will
              be returned as the actual
        """

        if road_id not in self._cached_map_model.roads_data:
            raise KeyError('road_id=%d is not in Map Model', road_id)

        relative_lon = start_lon + lon_step
        residual_lon = relative_lon
        # find road_id containing the target road_lon
        longitudes = self._cached_map_model.roads_data[road_id].longitudes
        road_length = longitudes[-1]
        while relative_lon > road_length:  # then advance to the next road
            relative_lon -= road_length
            residual_lon -= road_length
            next_road_id = navigation_plan.get_next_road(road_id, self.logger)
            if next_road_id is None:
                self.logger.warning(
                    "Couldn't achieve advantage of %f [m] on road_id %d. Terminated at with residual distance of %f",
                    (start_lon + lon_step), road_id, residual_lon)
                return road_id, road_length, residual_lon
            else:
                road_id = next_road_id
                longitudes = self._cached_map_model.roads_data[road_id].longitudes
                road_length = longitudes[-1]

        residual_lon -= relative_lon

        return road_id, relative_lon, residual_lon

    def _get_road_properties_in_world_coordinates(self, road_id, lon):
        # type: (int, float) -> (float, np.ndarray, np.ndarray)
        """
        :return: right-most road point at the given longitude with latitude vector (perpendicular to the local road's tangent),
            the first point index with longitude > the given longitude (for the next search)
            the longitude relatively to the next road (in case if the road_id has changed)
        """

        road_details = self._cached_map_model.roads_data[road_id]
        longitudes = road_details.longitudes
        max_longitude = longitudes[-1]
        if max_longitude < lon:
            error_message = ('asked for lon=%f out of max longitudes vector value (%f)', lon, max_longitude)
            self.logger.warning(error_message)
            raise Exception(error_message)
        points = road_details.points[0:2].transpose()
        width = road_details.width
        length = road_details.longitudes[-1]
        # get point with longitude > cur_lon
        pnt_ind = np.argmax(longitudes > lon)
        pnt_ind = max(1, pnt_ind)

        # calc lat_vec, right_point, road_lon of the target longitude.
        # the resulted road_id may differ from the input road_id because the target point may belong to another road.
        # for the same reason the resulted road_lon may differ from the input road_lon.
        lane_vec = self.normalize_vec(points[pnt_ind] - points[pnt_ind - 1])
        lat_vec = np.array([-lane_vec[1], lane_vec[0]])  # from right to left
        center_point = points[pnt_ind] - lane_vec * (longitudes[pnt_ind] - lon)
        right_point = center_point - lat_vec * (width / 2.)
        return length, right_point, lat_vec

    def convert_lat_lon_to_world(self, road_id, lat, lon, navigation_plan):
        # type: (int, float, float, NavigationPlanMsg) -> (np.ndarray, float)
        """
        Given road_id, lat & lon, calculate the point in world coordinates.
        Z coordinate is calculated using the OSM data: if road's head and tail are at different layers (height),
        then interpolate between them.
        :param road_id:
        :param lat:
        :param lon:
        :return: point in 3D world coordinates;
         actual_lon = lon - residual lon
        """

        # Get actual road_id and actual_lon, in case that the longitude exceeds road length
        actual_road_id, road_lon, residual_lon = \
            self._advance_road_coordinates_in_lon(road_id=road_id, start_lon=0.0,
                                                  lon_step=lon,
                                                  navigation_plan=navigation_plan)
        if actual_road_id != road_id:
            self.logger.info(
                'Conversion of (road=%d, lon=%f) exceeded max. lon. Returned point on road_id=%d',
                road_id, lon, actual_road_id)

        # Warn if couldn't supply sufficient lookahead
        actual_lon_lookahead = lon - residual_lon
        if not np.math.isclose(residual_lon, 0.0):
            self.logger.warning(
                'Conversion of (road=%d, lon=%f) terminated with actual lon of: %f',
                road_id, lon, actual_lon_lookahead)

        # Get road structure properties
        length, right_point, lat_vec = self._get_road_properties_in_world_coordinates(actual_road_id, road_lon)
        road_id = actual_road_id
        road_details = self._cached_map_model.roads_data[road_id]
        head_layer = road_details.head_layer
        tail_layer = road_details.tail_layer
        tail_wgt = lon / length
        z = head_layer * (1 - tail_wgt) + tail_layer * tail_wgt
        world_pnt = np.append(right_point + lat_vec * lat, [z])  # 3D point

        return world_pnt, actual_lon_lookahead

    def _convert_world_to_lat_lon_for_given_road(self, x, y, road_id):
        # type: (float, float, int) -> (float, float)
        """
        Convert point in world coordinates (x, y) to (lat, lon) of road with given road_id
        :param x: the point's world x coordinate in meters
        :param y: the point's world y coordinate in meters
        :param road_id:
        :return: signed lat (relatively to the road center), lon (from road start)
        """
        road_details = self._cached_map_model.roads_data[road_id]
        longitudes = road_details.longitudes
        # find the closest point of the road to (x,y)
        points = road_details.points[0:2].transpose()
        dist_2 = np.linalg.norm(np.asarray(points) - (x, y), axis=1)
        closest_pnt_ind = np.argmin(dist_2)  # index of the closest road point to (x,y)
        # find the closest segment and the distance (latitude)
        # given the closest road point, take two adjacent road segments around it and pick the closest segment to (x,y)
        # proj1, proj2 are projection points of p onto two above segments
        # lat_dist1, lat_dist2 are latitude distances from p to the segments
        # sign1, sign2 are the sign of the above two latitudes
        p = proj1 = proj2 = np.array([x, y])
        lat_dist1 = lat_dist2 = LARGE_NUM
        sign1 = sign2 = 0
        if closest_pnt_ind > 0:
            sign1, lat_dist1, proj1 = CartesianFrame.calc_point_segment_dist(p, points[closest_pnt_ind - 1],
                                                                             points[closest_pnt_ind])
        if closest_pnt_ind < len(points) - 1:
            sign2, lat_dist2, proj2 = CartesianFrame.calc_point_segment_dist(p, points[closest_pnt_ind],
                                                                             points[closest_pnt_ind + 1])
        if lat_dist1 < lat_dist2:
            lat_dist = lat_dist1
            sign = sign1
            lon = proj1 + longitudes[closest_pnt_ind - 1]
        else:
            lat_dist = lat_dist2
            sign = sign2
            lon = proj2 + longitudes[closest_pnt_ind]

        # Convert to location in road coordinates ([m] from rightmost edge of road):
        # lat_dist is measured relatively to the center of road,
        # and it's positive sign goes to the left side of the road
        lat = road_details.width / 2 + lat_dist * sign
        return lat, lon

    @staticmethod
    def normalize_vec(vec):
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
