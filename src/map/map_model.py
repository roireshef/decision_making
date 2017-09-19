from logging import Logger

import numpy as np
import copy
from typing import List, Dict

from decision_making.src.exceptions import RoadNotFound, MapCellNotFound
from decision_making.src.map.constants import Sidewalk


class RoadDetails:
    def __init__(self, id, name, points, head_node, tail_node,
                 head_layer, tail_layer, max_layer, lanes_num, oneway, lane_width,
                 sidewalk: Sidewalk, ext_head_yaw, ext_tail_yaw,
                 ext_head_lanes, ext_tail_lanes, turn_lanes):
        # type: (int, str, np.ndarray, int, int, int, int, int, int, bool, float, Sidewalk, float, float, int, int, List[str]) -> None
        """
        Road details class
        :param id: road's id
        :param name: road's name
        :param points: road's points array. numpy array of size Nx2 (N rows, 2 columns)
        :param head_node: node id of the road's head
        :param tail_node:
        :param head_layer: int layer of the road's head (0 means ground layer)
        :param tail_layer:
        :param max_layer: may be greater than head_layer & tail_layer, if the road's middle is a bridge
        :param lanes_num:
        :param oneway: true if the road is one-way
        :param lane_width: in meters
        :param sidewalk: may be 'left', 'right', 'both', 'none'
        :param ext_head_yaw: yaw of the incoming road
        :param ext_tail_yaw: yaw of the outgoing road
        :param ext_head_lanes: lanes number in the incoming road
        :param ext_tail_lanes: lanes number in the outgoing road
        :param turn_lanes: list of strings describing where each lane turns
        """
        assert points.shape[1] == 2, "points should be a Nx2 matrix"

        self._id = id
        self._name = name
        self._points = RoadDetails.remove_duplicate_points(points)
        self._longitudes = RoadDetails.calc_longitudes(self._points)
        self._head_node = head_node
        self._tail_node = tail_node
        self._head_layer = head_layer
        self._tail_layer = tail_layer
        self._max_layer = max_layer
        self._lanes_num = lanes_num
        self._oneway = oneway
        self._lane_width = lane_width
        if self._lanes_num is not None and self._lane_width is not None:
            self._road_width = self._lane_width * self._lanes_num
        self._sidewalk = sidewalk
        self._ext_head_yaw = ext_head_yaw
        self._ext_tail_yaw = ext_tail_yaw
        self._ext_head_lanes = ext_head_lanes
        self._ext_tail_lanes = ext_tail_lanes
        self._turn_lanes = turn_lanes

    @property
    def length(self):
        return self._longitudes[-1]

    @property
    def lanes_num(self):
        return self._lanes_num

    @property
    def lane_width(self):
        return self._lane_width

    @property
    def road_width(self):
        return self._road_width

    @staticmethod
    def remove_duplicate_points(points):
        # type: (np.ndarray) -> np.ndarray
        # TODO: move solution to mapping module
        return points[np.append(np.linalg.norm(np.diff(points, axis=0), axis=1) > 0.0, [True])]

    @staticmethod
    def calc_longitudes(points: np.ndarray) -> np.ndarray:
        """
        given road points, calculate array of longitudes of all points
        :param points: array of road points
        :return: longitudes array (longitudes[0] = 0)
        """
        points_direction = np.diff(np.array(points), axis=0)
        points_norm = np.linalg.norm(points_direction, axis=1)
        longitudes = np.concatenate(([0], np.cumsum(points_norm)))
        return longitudes

class MapModel:
    def __init__(self, roads_data, incoming_roads, outgoing_roads, xy2road_map, xy2road_tile_size):
        # type: (Dict[int, RoadDetails], Dict[int, List[int]], Dict[int, List[int]], Dict[(int, float, float), List[int]], float) -> None
        self._roads_data = copy.deepcopy(roads_data)  # dictionary: road_id -> RoadDetails
        self._incoming_roads = copy.deepcopy(incoming_roads)  # dictionary: node id -> incoming roads
        self._outgoing_roads = copy.deepcopy(outgoing_roads)  # dictionary: node id -> outgoing roads
        self._xy2road_map = copy.deepcopy(xy2road_map)  # maps world coordinates to road_ids
        self.xy2road_tile_size = xy2road_tile_size

    def get_road_data(self, road_id):
        # type: (int) -> RoadDetails
        try:
            return self._roads_data[road_id]
        except KeyError:
            raise RoadNotFound("MapModel doesn't have road {}".format(road_id))

    def get_road_ids(self):
        # type: () -> List[int]
        return list(self._roads_data.keys())

    def get_xy2road_cell(self, coordinates):
        # type: ((int, float, float)) -> List[int]
        try:
            return self._xy2road_map[coordinates]
        except KeyError:
            raise MapCellNotFound("MapModel's xy2road_map doesn't have cell {}".format(coordinates))


