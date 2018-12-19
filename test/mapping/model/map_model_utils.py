from typing import Dict, List, Tuple

import numpy as np
import pygame

from decision_making.src.mapping.model.constants import Sidewalk, EPSILON, ROADS_MAP_TILE_SIZE
from decision_making.src.mapping.model.map_model import MapModel, RoadDetails


class TestMapModelUtils:
    @staticmethod
    def create_road_map_from_coordinates(points_of_roads, road_id, road_name, lanes_num, lane_width, frame_origin=None):
        # type: (List[np.ndarray], List[int], List[str], List[int], List[float], List[float]) -> MapModel
        """
        :param points_of_roads: LIST OF - array of center road points (numpy array of size Nx2 (N rows, 2 columns)
        :param road_id: LIST OF - the initial road id to generate road ids from
        :param road_name: LIST OF - the name of the road (will be identical for all roads)
        :param lanes_num: LIST OF - the number of lanes in the roads (will be identical for all roads)
        :param lane_width: LIST OF - the width of a lane (will be identical for all lanes)
        :param frame_origin: 0,0 location of the map, in Geo-coordinates (lat, lon)
        :return: a map model generated according to the received parameters
        """
        roads_data = dict()
        xy2road_map = dict()
        incoming_roads = {0: []}
        outgoing_roads = {len(points_of_roads): []}

        for road_idx in range(len(points_of_roads)):
            # generate the road details
            road_details = RoadDetails(road_id[road_idx], road_name[road_idx], points_of_roads[road_idx],
                                       road_idx, road_idx+1, 0, 0, 0,
                                       lanes_num[road_idx], True, lane_width[road_idx],
                                       Sidewalk.NONE, 0, 0, 0, 0, [])

            roads_data[road_id[road_idx]] = road_details
            # Render the mapping of (x,y) -> road id
            xy2road_map = TestMapModelUtils.__render_road(road_id[road_idx],
                                            road_details._points,
                                            road_details.road_width,
                                            road_details._max_layer,
                                            xy2road_map=xy2road_map)

            incoming_roads[road_idx+1] = [road_id[road_idx]]
            outgoing_roads[road_idx] = [road_id[road_idx]]

        # Generate model
        map_model = MapModel(roads_data=roads_data, incoming_roads=incoming_roads, outgoing_roads=outgoing_roads,
                             xy2road_map=xy2road_map,
                             xy2road_tile_size=ROADS_MAP_TILE_SIZE, frame_origin=frame_origin)

        return map_model

    @staticmethod
    def split_road(map_model, parts_num):
        # type: (MapModel, int) -> MapModel
        if len(map_model.get_road_ids()) > 1:
            return map_model
        road_id = map_model.get_road_ids()[0]
        points = map_model.get_road_data(road_id)._points
        road_name = map_model.get_road_data(road_id)._name
        num_lanes = map_model.get_road_data(road_id).lanes_num
        lane_width = map_model.get_road_data(road_id).lane_width

        part_length = int(points.shape[0] / parts_num)
        split_points = []
        road_ids = []
        road_names = []
        for i in range(parts_num):
            split_points.append(points[i*part_length : (i+1)*part_length + 1]
                                if i < parts_num-1 else points[i*part_length:])
            road_ids.append(road_id + i)
            road_names.append(road_name + str(i))
        return TestMapModelUtils.create_road_map_from_coordinates(split_points, road_ids, road_names,
                                                                  [num_lanes]*parts_num, [lane_width]*parts_num)

    @staticmethod
    def __render_road(road_id, points, road_width, max_layer, road_margin=3,
                      road_map_tile_size=10, xy2road_map=dict()):
        # type: (int, np.ndarray, float, int, float, int, Dict[Tuple[int, float, float], List[int]])->dict
        """
        Render roads on xy2road_map, such that for every square tile_size we know in which road it's contained
        In addition, move road's points such that the lanes will fit to prev and next road with different number of lanes
        :param road_id:
        :param points: road center points (Nx2 numpy array)
        :param xy2road_map: the output map as a dictionary: (layer, x,y) -> road_ids list
        :return: updates the input variable xy2road_map with the new road and returns it
        """

        points = points.transpose()
        xmin = int(min(points[0, :]) - road_width - road_margin)
        xmax = int(max(points[0, :]) + road_width + road_margin + 1)
        ymin = int(min(points[1, :]) - road_width - road_margin)
        ymax = int(max(points[1, :]) + road_width + road_margin + 1)
        prev_point = points[:, 0]
        vertices = []
        for point in range(1, len(points[0, :])):
            cur_point = points[:, point]
            length = np.sqrt((cur_point[1] - prev_point[1]) ** 2 + (cur_point[0] - prev_point[0]) ** 2)
            if length < EPSILON:
                continue
            lat_vec = [(cur_point[1] - prev_point[1]) / length, -(cur_point[0] - prev_point[0]) / length]
            lat_vec = [lat_vec[0] * (0.5 * road_width + road_margin),
                       lat_vec[1] * (0.5 * road_width + road_margin)]
            if point == 1:
                vertices = [(prev_point[0] - lat_vec[0] - xmin, prev_point[1] - lat_vec[1] - ymin),
                            (prev_point[0] + lat_vec[0] - xmin, prev_point[1] + lat_vec[1] - ymin)]
            vertices = [(cur_point[0] - lat_vec[0] - xmin, cur_point[1] - lat_vec[1] - ymin)] + \
                       vertices + [(cur_point[0] + lat_vec[0] - xmin, cur_point[1] + lat_vec[1] - ymin)]
            prev_point = cur_point

        color = (255, 255, 255, 255)
        surface = pygame.Surface((xmax - xmin, ymax - ymin))
        pygame.draw.polygon(surface, color, vertices, 0)
        L = max_layer

        for y in range(ymax - ymin):
            for x in range(xmax - xmin):
                if surface.get_at((x, y))[0] > 0:
                    X = int(round((float(x + xmin)) / road_map_tile_size))
                    Y = int(round((float(y + ymin)) / road_map_tile_size))
                    if (L, X, Y) in xy2road_map:
                        if road_id not in xy2road_map[(L, X, Y)]:
                            xy2road_map[(L, X, Y)].append(road_id)
                    else:
                        xy2road_map[(L, X, Y)] = [road_id]

        return xy2road_map
