import copy
from typing import List, Dict, Tuple

import numpy as np

from decision_making.test.mapping.exceptions import RoadNotFound, MapCellNotFound, NextRoadNotFound, raises
from decision_making.test.mapping.model.constants import Sidewalk


'''
Naive Road-Level MapModel
'''
class RoadDetails(object):
    def __init__(self, id, name, points, head_node, tail_node,
                 head_layer, tail_layer, max_layer, lanes_num, oneway, lane_width,
                 sidewalk, ext_head_yaw, ext_tail_yaw,
                 ext_head_lanes, ext_tail_lanes, turn_lanes, points_downsample_step=1):
        # type: (int, str, np.ndarray, int, int, int, int, int, int, bool, float, Sidewalk, float, float, int, int, List[str], float) -> None
        """
        Road details class
        :param id: road's id
        :param name: road's name
        :param points: center road points array. numpy array of size Nx2 (N rows, 2 columns)
        :param head_node: node id of the road's head
        :param tail_node: node id of the road's tail
        :param head_layer: int layer of the road's head (0 means ground layer)
        :param tail_layer: the layer of the road's tail (0 means ground layer)
        :param max_layer: may be greater than head_layer & tail_layer, if the road's middle is a bridge
        :param lanes_num: the number of lanes in the road
        :param oneway: true if the road is one-way
        :param lane_width: the width of a lane [m]
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
            self._points = RoadDetails.downsample_points(points, points_downsample_step)
        else:
            self._points = RoadDetails.remove_duplicate_points(points)

        self._id = id
        self._name = name
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

    @property
    def tail_node(self):
        return self._tail_node

    @property
    def head_node(self):
        return self._head_node

    @staticmethod
    def remove_duplicate_points(points):
        # type: (np.ndarray) -> np.ndarray
        """
        Remove adjacent duplicate points
        :param points: numpy array of size Nx2 (N rows, 2 columns)
        :return: a numpy array of points without the duplicated points
        """
        return points[np.append(np.linalg.norm(np.diff(points, axis=0), axis=1) > 0.0, [True])]

    @staticmethod
    def calc_longitudes(points):
        # type: (np.ndarray) -> np.ndarray
        """
        given road points , calculate array of longitudes of all points
        :param points: array of road points (numpy array of size Nx2 (N rows, 2 columns)
        :return: longitudes array (length N), measured from the first point (longitudes[0] = 0)
        """
        points_direction = np.diff(np.array(points), axis=0)
        points_norm = np.linalg.norm(points_direction, axis=1)
        longitudes = np.concatenate(([0], np.cumsum(points_norm)))
        return longitudes

    @staticmethod
    def downsample_points(points, sample_step):
        # type: (np.ndarray, float) -> np.ndarray
        """
        Takes an array of points and returns a new array containing only points that are distanced enough from one
        another
        :param points: array of road points (numpy array of size Nx2 (N rows, 2 columns)
        :param sample_step: the step size of the sampling (tolerance distance)
        :return: a down sampled numpy array of points without the points that are distanced enough
        """
        downsampled = np.array([points[0]])
        for p in points:
            if np.linalg.norm(downsampled[-1] - p) >= sample_step:
                downsampled = np.vstack([downsampled, p])
        return downsampled


class MapModel(object):
    def __init__(self, roads_data, incoming_roads, outgoing_roads, xy2road_map, xy2road_tile_size, frame_origin):
        # type: (Dict[int, RoadDetails], Dict[int, List[int]], Dict[int, List[int]], Dict[Tuple[int, float, float], List[int]], float, List[float]) -> None
        """
        :param roads_data: dictionary: road_id -> RoadDetails
        :param incoming_roads: dictionary: node id -> incoming roads
        :param outgoing_roads: dictionary: node id -> outgoing roads
        :param xy2road_map: dictionary: cell grid coordinates (layer, x, y) to road_ids.
                            Layer is required since the same x,y point may cause layer ambiguity
        :param xy2road_tile_size: The size of the xy grid used to search for x,y to map conversion
        :param frame_origin: The frame origin - 0,0 location of the map, in Geo-coordinates (lat, lon)
        """

        self._roads_data = copy.deepcopy(roads_data)
        self._incoming_roads = copy.deepcopy(incoming_roads)
        self._outgoing_roads = copy.deepcopy(outgoing_roads)
        self._xy2road_map = copy.deepcopy(xy2road_map)
        self._frame_origin = frame_origin
        self.xy2road_tile_size = xy2road_tile_size

    @raises(RoadNotFound)
    def get_road_data(self, road_id):
        # type: (int) -> RoadDetails
        """
        :param road_id: the requested road id
        :return: the road's data
        """
        try:
            return self._roads_data[road_id]
        except KeyError:
            raise RoadNotFound("MapModel doesn't have road {}".format(road_id))

    @raises(RoadNotFound, NextRoadNotFound)
    def get_next_road(self, road_id):
        # type: (int) -> int
        """
        :param road_id: the requested road id
        :return: the road id of the next road
        """
        try:
            tail_node = self._roads_data[road_id].tail_node
        except:
            raise RoadNotFound("Trying to get the next road for road {} which doesn't exist in map".format(road_id))
        if tail_node in self._outgoing_roads and len(self._outgoing_roads[tail_node]) > 0:
            return self._outgoing_roads[tail_node][0]
        else:
            raise NextRoadNotFound("No road are connected to road {}".format(road_id))

    @raises(RoadNotFound, NextRoadNotFound)
    def get_prev_road(self, road_id):
        # type: (int) -> int
        """
        :param road_id: the requested road id
        :return: the road id of the previous road
        """
        try:
            head_node = self._roads_data[road_id].head_node
        except:
            raise RoadNotFound("Trying to get the next road for road {} which doesn't exist in map".format(road_id))
        if head_node in self._incoming_roads and len(self._incoming_roads[head_node]) > 0:
            return self._incoming_roads[head_node][0]
        else:
            raise NextRoadNotFound("No road are connected to road {}".format(road_id))

    def get_road_ids(self):
        # type: () -> List[int]
        """
        :return: a list of all the road ids in the map
        """
        return list(self._roads_data.keys())

    @raises(MapCellNotFound)
    def get_xy2road_cell(self, coordinates):
        # type: ((int, float, float)) -> List[int]
        """
        :param coordinates: a tuple of (layer, x, y), where x,y are cell grid coordinates.
        :return: the list of road ids within the cell grid
        """
        try:
            return self._xy2road_map[coordinates]
        except KeyError:
            raise MapCellNotFound("MapModel's xy2road_map doesn't have cell {}".format(coordinates))
