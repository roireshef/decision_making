from typing import Dict

import numpy as np
import pygame

from decision_making.src.map.constants import Sidewalk, EARTH_RADIUS, EPSILON, ROADS_MAP_TILE_SIZE
from decision_making.src.map.map_model import MapModel, RoadDetails


class TestMapModelUtils:
    @staticmethod
    def create_road_map_from_coordinates(points_of_roads, road_id, road_name, lanes_num, lane_width):
        roads_data = dict()
        xy2road_map = dict()
        for road_ind, points in enumerate(points_of_roads):

            road_gen_id = road_id + road_ind

            # build cumulative longitudes list for the road points
            longitudes = TestMapModelUtils.__calc_longitudes(points)

            # generate the road details
            road_details = RoadDetails(road_gen_id, road_name, points, longitudes, 0, 0, 0, 0, 0, lanes_num, True, lane_width,
                                       Sidewalk.NONE, 0, 0, 0, 0, [])

            roads_data[road_gen_id] = road_details
            # Render the mapping of (x,y) -> road id
            xy2road_map = TestMapModelUtils.__render_road(road_gen_id,
                                                          road_details.points,
                                                          road_details.width,
                                                          road_details.max_layer,
                                                          xy2road_map=xy2road_map)

        # Generate model
        map_model = MapModel(roads_data=roads_data, incoming_roads={}, outgoing_roads={}, xy2road_map=xy2road_map,
                             xy2road_tile_size=ROADS_MAP_TILE_SIZE)

        return map_model

    @staticmethod
    def __convert_coordinates_to_frame(coords: np.ndarray, layer: int) -> np.ndarray:
        """
        convert points from earth coordinates to the model frame
        :param coords: array of points in original earth coordinates (lat/lon)
        :param layer: 0 ground level, 1 bridge, 2 bridge above bridge, etc
        :return: points in model frame with respect to the starting coordinates
        """
        if not coords.any():
            return np.array([])

        lats = coords[:, 0]
        lons = coords[:, 1]

        lat2 = np.radians(lats)
        lon2 = np.radians(lons)

        lat_start = np.min(lats)
        lon_start = np.min(lons)

        # earth coordinates relatively to the local submap
        dlat = lat2 - np.radians(lat_start)
        dlon = lon2 - np.radians(lon_start)

        a = (np.sin(dlat / 2) * np.sin(dlat / 2) +
             np.sin(dlon / 2) * np.sin(dlon / 2) * np.cos(np.radians(lat_start)) * np.cos(lat2))
        distances = 2 * EARTH_RADIUS * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        angles = (np.arctan2(np.sin(dlon) * np.cos(lat2),
                             np.cos(np.radians(lat_start)) *
                             np.sin(lat2) -
                             np.sin(np.radians(lat_start)) *
                             np.cos(lat2) * np.cos(dlon)))

        # array of points in 3D world coordinates (x,y,layer)
        points = np.array([distances * np.sin(angles) * 1000,
                           distances * np.cos(angles) * 1000,
                           np.repeat(float(layer), np.shape(distances)[0])])

        return lat_start, lon_start, points

    @staticmethod
    def __calc_longitudes(points: np.ndarray) -> np.ndarray:
        """
        given road points, calculate array of longitudes of all points
        :param points: array of road points
        :return: longitudes array (longitudes[0] = 0)
        """
        points_direction = np.diff(np.array(points), axis=0)
        points_norm = np.linalg.norm(points_direction, axis=1)
        longitudes = np.concatenate(([0], np.cumsum(points_norm)))
        return longitudes

    @staticmethod
    def __render_road(road_id: int, points: np.ndarray, road_width: float, max_layer: int, road_margin: float = 3,
                      road_map_tile_size: int = 10, xy2road_map: Dict = {}):
        """
        Render roads on xy2road_map, such that for every square tile_size we know in which road it's contained
        In addition, move road's points such that the lanes will fit to prev and next road with different number of lanes
        :param road_id:
        :param points: road center points (Nx2 numpy array)
        :param xy2road_map: the output map as a dictionary: (x,y) -> road_ids list
        :return: xy2road_map: the output map as a dictionary: (x,y) -> road_ids list
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
        #xy2road_map = {}
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
