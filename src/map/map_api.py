import numpy as np

from decision_making.src.map.constants import *
from decision_making.src.map.map_model import MapModel
from typing import List, Union
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.planning.utils.geometry_utils import CartesianFrame


class RoadDetails:
    def __init__(self, id, name, points, longitudes, head_node, tail_node,
                 head_layer, tail_layer, max_layer, lanes_num, one_way, lane_width,
                 side_walk: Sidewalk, ext_head_yaw, ext_tail_yaw,
                 ext_head_lanes, ext_tail_lanes, turn_lanes):
        # type: (int, str, np.ndarray, np.ndarray, int, int, int, int, int, int, bool, float, Sidewalk, float, float, int, int, List[str]) -> None
        """
        Road details class
        :param id: road's id
        :param name: road's name
        :param points: road's points array. numpy array of size 2xN (2 rows, N columns)
        :param longitudes: list of longitudes of the road's points starting from 0
        :param head_node: node id of the road's head
        :param tail_node:
        :param head_layer: int layer of the road's head (0 means ground layer)
        :param tail_layer:
        :param max_layer: may be greater than head_layer & tail_layer, if the road's middle is a bridge
        :param lanes_num:
        :param one_way: true if the road is one-way
        :param lane_width: in meters
        :param side_walk: may be 'left', 'right', 'both', 'none'
        :param ext_head_yaw: yaw of the incoming road
        :param ext_tail_yaw: yaw of the outgoing road
        :param ext_head_lanes: lanes number in the incoming road
        :param ext_tail_lanes: lanes number in the outgoing road
        :param turn_lanes: list of strings describing where each lane turns
        """
        self.id = id
        self.name = name
        self.points = points
        self.longitudes = longitudes
        self.head_node = head_node
        self.tail_node = tail_node
        self.head_layer = head_layer
        self.tail_layer = tail_layer
        self.max_layer = max_layer
        self.lanes_num = lanes_num
        self.one_way = one_way
        self.lane_width = lane_width
        self.width = lane_width*lanes_num
        self.side_walk = side_walk
        self.ext_head_yaw = ext_head_yaw
        self.ext_tail_yaw = ext_tail_yaw
        self.ext_head_lanes = ext_head_lanes
        self.ext_tail_lanes = ext_tail_lanes
        self.turn_lanes = turn_lanes


class MapAPI:
    def __init__(self, map_model):
        # type: (MapModel) -> None
        self._cached_map_model = map_model
        pass

    def find_roads_containing_point(self, layer, world_x, world_y):
        # type: (int, float, float) -> List[int]
        """
        shortcut to a cell of the map xy2road_map
        :param layer: 0 ground, 1 on bridge, 2 bridge above bridge, etc
        :param world_x: world coordinates in meters
        :param world_y: world coordinates in meters
        :return: road_ids containing the point (world_x, world_y)
        """
        cell_x = int(round(world_x / ROADS_MAP_TILE_SIZE))
        cell_y = int(round(world_y / ROADS_MAP_TILE_SIZE))
        return self._cached_map_model.xy2road_map.get((layer, cell_x, cell_y), default=[])

    def get_center_lanes_latitudes(self, road_id):
        # type: (int) -> np.array
        """
        get list of latitudes of all lanes in the road
        :param road_id:
        :return: list of latitudes of all lanes in the road
        """
        road_details = self._cached_map_model.roads_data[road_id]
        lanes_num = road_details.lanes_num
        road_width = road_details.width
        lane_width = float(road_width) / lanes_num
        center_lanes = lane_width / 2 + np.array(range(lanes_num)) * lane_width
        return center_lanes

    def get_road_main_details(self, road_id):
        # type: (int) -> (int, float, float, np.ndarray)
        """
        get details of a given road
        :param road_id:
        :return: lanes number, road width, road length, road's points
        """
        if road_id not in self._cached_map_model.roads_data.keys():
            return None, None, None, None
        road_details = self._cached_map_model.roads_data[road_id]
        return road_details.lanes_num, road_details.width, road_details.longitudes[-1], road_details.points

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
        if z > 1:
            road_ids = self.find_roads_containing_point(1, x, y)
        else:
            road_ids = self.find_roads_containing_point(0, x, y)
            if road_ids is None:
                road_ids = self.find_roads_containing_point(1, x, y)
        if road_ids is None or len(road_ids) == 0:
            return None, None, None, None, None, None

        # find the closest road to (x,y) among the road_ids list
        (lat_dist, sign, lon, road_yaw, road_id) = self.__find_closest_road(x, y, road_ids)

        road_details = self._cached_map_model.roads_data[road_id]
        lanes_num = road_details.lanes_num
        lane_width = road_details.width / float(lanes_num)

        # calc lane number, intra-lane lat and yaw
        full_lat = lat_dist * sign + 0.5 * lanes_num * lane_width  # latitude relatively to the right road edge
        lane = float(int(full_lat / lane_width))  # from right to left
        lane = np.clip(lane, 0, lanes_num - 1)
        yaw_in_road = (yaw - road_yaw + 2 * np.pi) % (2 * np.pi)
        lane_lat = full_lat % lane_width
        return road_id, lane, full_lat, lane_lat, lon, yaw_in_road

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
        if to_road_id == from_road_id:  # simple case
            return to_lon_in_road - from_lon_in_road, True

        road_index_in_plan = navigation_plan.get_road_index_in_plan(from_road_id)
        if road_index_in_plan is None:  # target road_id not found in the navigation plan
            return max_lookahead_distance, False

        found_connection = False
        total_lon_distance = max_lookahead_distance

        # first search forward (direction=1); if not found then search backward (direction=-1)
        for direction in range(1, -2, -2):

            # 1. First road segment
            if direction > 0:  # forward
                total_lon_distance = self._cached_map_model.roads_data[from_road_id].longitudes[-1] - from_lon_in_road
            else:  # backward
                total_lon_distance = from_lon_in_road

            # 2. Middle road segments
            road_id, road_index_in_plan = navigation_plan.get_next_road(direction, road_index_in_plan)
            while road_id is not None and road_id != to_road_id and total_lon_distance < max_lookahead_distance:
                road_length = self._cached_map_model.roads_data[road_id].longitudes[-1]
                total_lon_distance += road_length
                road_id, road_index_in_plan = navigation_plan.get_next_road(direction, road_index_in_plan)

            # 3. Add length of last road segment
            if road_id == to_road_id:
                if direction > 0:  # forward
                    total_lon_distance += to_lon_in_road
                else:  # backward
                    total_lon_distance += self._cached_map_model.roads_data[to_road_id].longitudes[-1] - to_lon_in_road
                found_connection = True
                break  # stop the search when the connection is found

        return total_lon_distance, found_connection

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
        if road_id is None:
            road_index_in_plan = navigation_plan.current_road_index
        else:
            road_index_in_plan = navigation_plan.get_road_index_in_plan(road_id)

        road_details = self._cached_map_model.roads_data[road_id]
        road_width = road_details.width
        center_road_lat = road_width / 2.0

        longitudes = road_details.longitudes
        path = np.zeros(shape=[2, 0])

        road_length = road_details.longitudes[-1]
        closest_longitude_index = np.argmin(np.abs(longitudes - lon))

        # Init with current road
        road_starting_longitude = lon
        residual_lookahead = max_lookahead_distance

        path_points = road_details.points
        if direction == 1:
            target_longitude_index = np.argmin(np.abs(longitudes - (residual_lookahead + road_starting_longitude)))
            first_exact_lon_point = self._convert_lat_lon_to_world(road_id, center_road_lat, lon, navigation_plan)
            if first_exact_lon_point is None:
                return None
            partial_path_points = path_points[:, closest_longitude_index:target_longitude_index + 1]
            partial_path_points[:, 0] = first_exact_lon_point[0:2]
            achieved_lookahead = road_length - lon
        else:
            target_longitude_index = np.argmin(np.abs((road_length - longitudes) - residual_lookahead))
            first_exact_lon_point = self._convert_lat_lon_to_world(road_id, center_road_lat, lon, navigation_plan)
            if first_exact_lon_point is None:
                return None
            partial_path_points = path_points[:, target_longitude_index:closest_longitude_index + 1]
            partial_path_points = np.flip(partial_path_points, axis=1)
            partial_path_points[:, 0] = first_exact_lon_point[0:2]
            achieved_lookahead = lon

        path = np.concatenate((path, partial_path_points), axis=1)
        # Iterate over next road, until we get enough lookahead
        while achieved_lookahead < max_lookahead_distance and road_id is not None:
            road_starting_longitude = 0
            road_id, road_index_in_plan = navigation_plan.get_next_road(direction, road_index_in_plan)

            if road_id is not None:
                road_details = self._cached_map_model.roads_data[road_id]
                longitudes = road_details.longitudes
                road_length = longitudes[-1]
                path_points = road_details.points

                residual_lookahead = max_lookahead_distance - achieved_lookahead
                if road_length > residual_lookahead:
                    # Take only the relevant part of the current road
                    if direction == 1:
                        target_longitude_index = np.argmin(
                            np.abs(longitudes - (residual_lookahead + road_starting_longitude)))
                        partial_path_points = path_points[:, :target_longitude_index + 1]
                    else:
                        target_longitude_index = np.argmin(np.abs((road_length - longitudes) - residual_lookahead))
                        partial_path_points = path_points[:, target_longitude_index:]
                        partial_path_points = np.flip(partial_path_points, axis=1)

                else:
                    # Take whole road, because there is more ground to cover
                    partial_path_points = path_points

            from_idx = 0
            d_xy = path[:, -1] - partial_path_points[:, 0]
            if np.sum(d_xy ** 2) == 0:  # avoid duplicated start point of the next path
                from_idx = 1
            path = np.concatenate((path, partial_path_points[:, from_idx:]), axis=1)
            achieved_lookahead += road_length

        # Replace the last (closest, but inexact) point, and replace it with a point with the exact lon value
        if direction == 1:
            lon = residual_lookahead + road_starting_longitude
        else:
            lon = road_length - (road_starting_longitude + residual_lookahead)
        last_exact_lon_point = self._convert_lat_lon_to_world(road_id, center_road_lat, lon, navigation_plan)
        if last_exact_lon_point is None:
            return None

        # if path consists of a single point, add it to the end of route. else, replace last point
        path_length = path.shape[1]
        if path_length > 1:
            path[:, -1] = last_exact_lon_point[0:2]
        else:
            d_xy = path[:, -1] - last_exact_lon_point
            if np.sum(d_xy ** 2) == 0:  # avoid duplicated point of the next path
                path = np.concatenate((path, last_exact_lon_point[0:2].reshape([2, 1])), axis=1)

        if lat != 0:
            shift_amount = -road_width / 2.0 + lat
            lat_shifted_path = self._shift_road_vector_in_lat(points=path, lat_shift=shift_amount)
        else:
            lat_shifted_path = path

        return lat_shifted_path

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
        shifted_path = self.get_path_lookahead(road_id, starting_lon, lat, lon_step*steps_num, navigation_plan, direction=1)
        resampled_path, _ = CartesianFrame.resample_curve(shifted_path.transpose(), lon_step)
        return resampled_path.transpose()

    def update_perceived_roads(self):
        pass

    def __find_closest_road(self, x, y, road_ids):
        # type: (float, float, List[int]) -> (float, float, float, float, float)
        # find the closest road to (x,y)
        closest_lat = LARGE_NUM
        closest_id = closest_sign = closest_yaw = closest_lon = None
        for road_id in road_ids:
            sign, lat_dist, lon, road_vec = self._convert_world_to_lat_lon_for_given_road(x, y, road_id)
            if lat_dist < closest_lat:
                road_yaw = np.arctan2(road_vec[1], road_vec[0])
                (closest_lat, closest_sign, closest_lon, closest_yaw, closest_id) = \
                    (lat_dist, sign, lon, road_yaw, road_id)
        return (closest_lat, closest_sign, closest_lon, closest_yaw, closest_id)

    @staticmethod
    def _shift_road_vector_in_lat(points, lat_shift):
        # type: (np.ndarray, float) -> np.ndarray
        """
        Given points list along a road, shift them laterally by lat_shift meters
        :param points (2xN): points list along a given road
        :param lat_shift: shift in meters
        :return: shifted points array (2xN)
        """
        pass

    def _convert_lon_to_world(self, road_id, pnt_ind, road_lon, navigation_plan, road_index_in_plan=None):
        # type: (int, int, float, NavigationPlanMsg, int) -> np.ndarray
        """
        Calculate world point matching to a given longitude of a given road.
        If the given longitude exceeds the current road length, then calculate the point in the next road.
        The next road is picked from the navigation plan.
        :param road_id: current road_id
        :param pnt_ind: index of the road point, from which we can search the new point (for speed optimization)
        :param road_lon: the point's longitude relatively to the start of the road_id
        :param navigation_plan: of type NavigationPlan, includes list of road_ids
        :param road_index_in_plan: current index in the navigation plan
        :return: new road_id (maybe the same one) and its length,
            right-most road point at the given longitude with latitude vector (perpendicular to the local road's tangent),
            the first point index with longitude > the given longitude (for the next search)
            the longitude relatively to the next road (in case if the road_id has changed)
            the navigation plan index.
            the resulted road_id may differ from the input road_id because the target point may belong to another road.
            for the same reason the resulted road_lon may differ from the input road_lon.
        """
        pass

    def _convert_lat_lon_to_world(self, road_id, lat, lon, navigation_plan):
        # type: (int, float, float, NavigationPlanMsg) -> np.ndarray
        """
        Given road_id, lat & lon, calculate the point in world coordinates.
        Z coordinate is calculated using the OSM data: if road's head and tail are at different layers (height),
        then interpolate between them.
        :param road_id:
        :param lat:
        :param lon:
        :return: point in 3D world coordinates
        """
        pass

    def _convert_world_to_lat_lon_for_given_road(self, x, y, road_id):
        # type: (float, float, int) -> (int, float, float, np.ndarray)
        """
        calc lat, lon, road dir for point=(x,y) and a given road
        :param x: the point's world x coordinate in meters
        :param y: the point's world y coordinate in meters
        :param road_id:
        :return: signed lat (relatively to the road center), lon (from road start), road_vec
        """
        pass
