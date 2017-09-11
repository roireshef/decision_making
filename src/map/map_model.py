from logging import Logger

import numpy as np
import copy
from typing import List, Dict

from decision_making.src.exceptions import RoadNotFound, MapCellNotFound
from decision_making.src.map.constants import Sidewalk


class RoadDetails:
    def __init__(self, id, name, points, longitudes, head_node, tail_node,
                 head_layer, tail_layer, max_layer, lanes_num, oneway, lane_width,
                 sidewalk: Sidewalk, ext_head_yaw, ext_tail_yaw,
                 ext_head_lanes, ext_tail_lanes, turn_lanes, points_downsample_step=0):
        # type: (int, str, np.ndarray, np.ndarray, int, int, int, int, int, int, bool, float, Sidewalk, float, float, int, int, List[str], float) -> None
        """
        Road details class
        :param id: road's id
        :param name: road's name
        :param points: road's points array. numpy array of size Nx2 (N rows, 2 columns)
        :param longitudes: list of longitudes of the road's points starting from 0
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
        :param points_downsample_step: whether to perform downsampling of the points with given step
        """
        assert points.shape[1] == 2, "points should be a Nx2 matrix"
        if points_downsample_step > 0:
            self.points = self.downsample_points(points, points_downsample_step)
        else:
            self.points = points

        self.id = id
        self.name = name
        self.longitudes = longitudes
        self.head_node = head_node
        self.tail_node = tail_node
        self.head_layer = head_layer
        self.tail_layer = tail_layer
        self.max_layer = max_layer
        self.lanes_num = lanes_num
        self.oneway = oneway
        self.lane_width = lane_width
        if lanes_num is not None and lane_width is not None:
            self.width = lane_width*lanes_num
        self.sidewalk = sidewalk
        self.ext_head_yaw = ext_head_yaw
        self.ext_tail_yaw = ext_tail_yaw
        self.ext_head_lanes = ext_head_lanes
        self.ext_tail_lanes = ext_tail_lanes
        self.turn_lanes = turn_lanes

    @property
    def length(self):
        return self.longitudes[-1]

    @staticmethod
    def downsample_points(points, sample_step):
        downsampled = np.array([points[0]])
        for p in points:
            if np.linalg.norm(downsampled[-1] - p) >= sample_step:
                downsampled = np.vstack([downsampled, p])
        return downsampled


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

    @staticmethod
    def remove_duplicate_points(points):
        # type: (np.ndarray) -> np.ndarray
        return points[np.append(np.sum(np.diff(points, axis=0), axis=1) != 0.0, [True])]
