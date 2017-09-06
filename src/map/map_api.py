from abc import ABCMeta
from logging import Logger
from typing import List

import numpy as np
import six

from decision_making.src.exceptions import *
from decision_making.src.exceptions import RoadNotFound, raises, LongitudeOutOfRoad, MapCellNotFound
from decision_making.src.global_constants import *
from decision_making.src.map.map_model import MapModel
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.planning.utils.geometry_utils import CartesianFrame, Euclidean


@six.add_metaclass(ABCMeta)
class MapAPI:
    def __init__(self, map_model, logger):
        # type: (MapModel, Logger) -> None
        self._cached_map_model = map_model
        self.logger = logger

    def update_perceived_roads(self):
        pass

    '''####################'''
    ''' EXTERNAL FUNCTIONS '''
    '''####################'''

    @raises(MapCellNotFound, RoadNotFound, LongitudeOutOfRoad)
    def convert_global_to_road_coordinates(self, x, y):
        # type: (float, float) -> (int, float, float, bool)
        """
        Convert a point in global coordinate frame to road coordinate, by searching for the nearest road and
        projecting it onto this road
        :param x: x coordinate in global coordinate frame
        :param y: y coordinate in global coordinate frame
        :return: Road ID, longitude from the road's start, latitude **relative to road's right-side**,
            is object within road latitudes
        """
        relevant_road_ids = self._find_roads_containing_point(x, y)
        closest_road_id = self._find_closest_road(x, y, relevant_road_ids)

        lon, lat = self._convert_global_to_road_coordinates(x, y, closest_road_id)

        road_width = self._get_road(closest_road_id).width
        is_on_road = bool(0.0 <= lat <= road_width)

        return closest_road_id, lon, lat, is_on_road

    @raises(RoadNotFound)
    def get_center_lanes_latitudes(self, road_id):
        # type: (int) -> np.ndarray
        """
        Get list of latitudes of all centers of lanes in the road
        :param road_id: Road ID to iterate over its lanes
        :return: list of latitudes of all centers of lanes in the road relative to the right side of the road
        """
        road_details = self._get_road(road_id)
        lanes_num = road_details.lanes_num
        road_width = road_details.width
        lane_width = float(road_width) / lanes_num
        center_lanes = np.array(range(lanes_num)) * lane_width + lane_width / 2
        return center_lanes

    @raises(RoadNotFound)
    def get_longitudinal_difference(self, initial_road_id, initial_lon, final_road_id, final_lon, navigation_plan):
        # type: (int, float, int, float, NavigationPlanMsg) -> float
        """
        This function calculates the total longitudinal difference in [m] between two points in road coordinates.
        IMPORTANT: If the destination road id is not in the navigation plan, raises RoadNotFound EXCEPTION.
         Handling needs to be done in the caller
        :param initial_road_id: initial road id (int)
        :param initial_lon: initial road longitude in [m]
        :param final_road_id: destination road id (int)
        :param final_lon: destination longitude in [m]
        :param navigation_plan: navigation plan according to which we advance on road
        :return: longitudinal difference in [m]
        """
        initial_road_idx = navigation_plan.get_road_index_in_plan(initial_road_id)
        final_road_idx = navigation_plan.get_road_index_in_plan(final_road_id)
        # look ahead
        if final_road_idx > initial_road_idx or (final_road_idx == initial_road_idx and final_lon > initial_lon):
            roads_ids = navigation_plan.road_ids[initial_road_idx:final_road_idx]  # this excludes last road
            roads_len = [self._get_road(rid).length for rid in roads_ids]
            return np.add(np.sum(roads_len), -initial_lon + final_lon)
        # look back
        else:
            roads_ids = navigation_plan.road_ids[final_road_idx:initial_road_idx]  # this excludes last road
            roads_len = [self._get_road(rid).length for rid in roads_ids]
            return -1 * np.add(np.sum(roads_len), - final_lon + initial_lon)

    @raises(RoadNotFound, LongitudeOutOfRoad)
    def advance_on_plan(self, initial_road_id, initial_lon, lookahead_dist, navigation_plan):
        # type: (int, float, float, NavigationPlanMsg) -> (int, float)
        """
        Given a longitude on specific road (<initial_road_id> and <initial_lon>), advance (lookahead) <desired_lon>
        distance. The lookahead iterates over the next roads specified in the <navigation_plan> and returns: (the final
        road id, the longitude along this road). If <desired_lon> is more than the distance to end of the plan, a
        LongitudeOutOfRoad exception is thrown.
        :param initial_road_id: the initial road_id (the vehicle is current on)
        :param initial_lon: initial longitude along <initial_road_id>
        :param lookahead_dist: the desired distance of lookahead in [m].
        :param navigation_plan: the relevant navigation plan to iterate over its road IDs.
        :return: (road_id, longitudinal distance from the beginning of <road_id>)
        """
        current_road_idx_in_plan = navigation_plan.get_road_index_in_plan(initial_road_id)
        roads_ids = navigation_plan.road_ids[current_road_idx_in_plan:]
        roads_len = [self._get_road(rid).length for rid in roads_ids]

        # distance to roads-ends
        roads_dist_to_end = np.cumsum(np.append([roads_len[0] - initial_lon], roads_len[1:]))
        # how much of lookahead_dist is left after this road
        roads_leftovers = np.subtract(lookahead_dist, roads_dist_to_end)

        try:
            target_road_idx = np.where(roads_leftovers < 0)[0][0]
            return roads_ids[target_road_idx], roads_leftovers[target_road_idx] + roads_len[target_road_idx]
        except IndexError:
            raise LongitudeOutOfRoad("The specified navigation plan is short {} meters to advance }{ in longitude"
                                     .format(roads_leftovers[-1], lookahead_dist))

    @raises(RoadNotFound)
    def advance_to_end_of_plan(self, initial_road_id, initial_lon, navigation_plan):
        # type: (int, float, NavigationPlanMsg) -> (int, float, float)
        """
        Given a longitude on specific road (<initial_road_id> and <initial_lon>), advance (lookahead) to the final point
        in the navigation plan.
        :param initial_road_id: the initial road_id (the vehicle is current on)
        :param initial_lon: initial longitude along <initial_road_id>
        :param navigation_plan: the relevant navigation plan to iterate over its road IDs.
        :return: (the last road id, its length, total distance to its end point)
        """
        initial_road_idx_in_plan = navigation_plan.get_road_index_in_plan(initial_road_id)
        roads_ids = navigation_plan.road_ids[initial_road_idx_in_plan:]
        roads_len = [self._get_road(rid).length for rid in roads_ids]
        roads_dist_to_end = np.sum(np.append([roads_len[0] - initial_lon], roads_len[1:]))  # dist to roads-ends
        return roads_ids[-1], roads_len[-1], roads_dist_to_end

    @raises(RoadNotFound, LongitudeOutOfRoad)
    def get_lookahead_points(self, initial_road_id, initial_lon, lookahead_dist, desired_lat, navigation_plan):
        # type: (int, float, float, float, NavigationPlanMsg) -> np.ndarray
        """
        Given a longitude on specific road, return all the points along this (and next) road(s) until reaching
        a lookahead of exactly <desired_lon> meters ahead. In addition, shift all points <desired_lat_shift> laterally,
        relative to the roads right-side.
        :param initial_road_id: the initial road_id (the vehicle is current on)
        :param initial_lon: initial longitude along <initial_road_id>
        :param lookahead_dist: the desired distance of lookahead in [m].
        :param desired_lat: desired lateral shift of points **relative to road's right-side**
        :param navigation_plan: the relevant navigation plan to iterate over its road IDs.
        :return:
        """
        # find the final point's (according to desired lookahead distance) road_id and longitude along this road
        final_road_id, final_lon = self.advance_on_plan(initial_road_id, initial_lon, lookahead_dist,
                                                        navigation_plan)
        initial_road_idx = navigation_plan.get_road_index_in_plan(initial_road_id)
        final_road_idx = navigation_plan.get_road_index_in_plan(final_road_id)
        relevant_road_ids = navigation_plan.road_ids[initial_road_idx:(final_road_idx + 1)]

        # exact projection of the initial point and final point on the road
        initial_point = self._convert_road_to_global_coordinates(initial_road_id, initial_lon, desired_lat)[0][:2]
        final_point = self._convert_road_to_global_coordinates(final_road_id, final_lon, desired_lat)[0][:2]

        # shift points (laterally) and concatenate all points of all relevant roads
        shifted_points = np.concatenate([self._shift_road_points_to_latitude(rid, desired_lat)
                                         for rid in relevant_road_ids])

        # calculate accumulate longitudinal distance for all points
        longitudes = np.cumsum(np.concatenate([np.append([0], np.diff(self._get_road(rid).longitudes))
                                               for rid in relevant_road_ids]))

        # trim shifted points from both sides according to initial point and final (desired) point
        shifted_points = shifted_points[np.greater(longitudes - initial_lon, 0) &
                                        np.less(longitudes - initial_lon, lookahead_dist)]

        # Build path
        path = np.concatenate(([initial_point], shifted_points, [final_point]))

        # Remove duplicate points (start of next road == end of last road)
        path = path[np.append(np.sum(np.diff(path, axis=0), axis=1) != 0.0, [True])]

        return path

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
        shifted = self.get_lookahead_points(road_id, starting_lon, lon_step * steps_num, lat_shift, navigation_plan)
        # TODO change to precise resampling
        resampled, _ = CartesianFrame.resample_curve(shifted, lon_step)
        return resampled

    '''####################'''
    ''' INTERNAL FUNCTIONS '''
    '''####################'''

    @raises(RoadNotFound)
    def _get_road(self, road_id):
        return self._cached_map_model.get_road_data(road_id)

    @raises(MapCellNotFound)
    def _find_roads_containing_point(self, x, y):
        # type: (float, float) -> List[int]
        """
        Returns the list of corresponding road IDs to a coordinate in the global-frame (x, y)
        :param x: world coordinates in meters
        :param y: world coordinates in meters
        :return: road_ids containing the point (x, y)
        """
        # TODO: unify cell-from-xy computation with the one in the map's creation procedure,
        tile_size = self._cached_map_model.xy2road_tile_size
        cell_x = int(round(x / tile_size))
        cell_y = int(round(y / tile_size))
        return self._cached_map_model.get_xy2road_cell((DEFAULT_MAP_LAYER, cell_x, cell_y))

    @raises(RoadNotFound)
    def _find_closest_road(self, x, y, road_ids):
        # type: (float, float, List[int]) -> int
        """
        Returns the closest road_id of the road which is closest to a point in the world (x, y).
        :param x: x coordinate on map (given in [m])
        :param y: y coordinate on map (given in [m])
        :param road_ids: list of road IDs to try to project the point on
        :return: road_id of the closest road
        """
        distances = [self._dist_to_road(x, y, rid) for rid in road_ids]
        return road_ids[np.argmin(distances)]

    @raises(RoadNotFound)
    def _dist_to_road(self, x, y, road_id):
        # type: (float, float, int) -> float
        """
        Compute distance to road by looking for its nearest point to (x,y), and computing the distance-to-segment to
        the segment before and after the nearest point
        :param x: x coordinate on map (given in [m])
        :param y: y coordinate on map (given in [m])
        :param road_id:
        :return: road ID to compute the distance to
        """
        point = np.array([x, y])
        road = self._get_road(road_id)

        # find the closest point of the road to (x,y)
        points = road.points[:, 0:2]
        distance_to_road_points = np.linalg.norm(np.array(points) - point, axis=0)
        closest_point_ind = np.argmin(distance_to_road_points)

        # the point (x,y) should be projected either onto the segment before the closest point or onto the one after it.
        closest_point_idx_pairs = [[closest_point_ind - 1, closest_point_ind],
                                   [closest_point_ind, closest_point_ind + 1]]

        # filter out non-existing indices
        closest_point_idx_pairs = closest_point_idx_pairs[np.greater_equal(closest_point_idx_pairs[:, 0], 0.0) &
                                                          np.less(closest_point_idx_pairs[:, 1], len(points))]

        segments_dists = [Euclidean.dist_to_segment_2d(point, points[pair[0]], points[pair[1]])
                          for pair in closest_point_idx_pairs]

        return min(segments_dists)

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
        road = self._get_road(road_id)
        points_with_yaw = CartesianFrame.add_yaw(road.points)

        if road.longitudes[0] <= lon <= road.longitudes[-1]:
            point_ind = np.where(road.longitudes <= lon)[0][-1]  # find index closest to target lon
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

    @raises(RoadNotFound, LongitudeOutOfRoad)
    def _convert_global_to_road_coordinates(self, x, y, road_id):
        # type: (float, float, int) -> (float, float)
        """
        Convert point in world coordinates (x, y) to (lat, lon) of road with given road_id
        If the point is on the road (in the sense of longitude), then lat is also the distance between the point
        and the road. Otherwise lat is the distance but not latitude, because in this case latitude is meaningless.
        :param x: the point's world x coordinate in meters
        :param y: the point's world y coordinate in meters
        :param road_id: road ID as in the map model
        :return: longitude (from road start), latitude **relative to road's right side**
        """
        point = np.array([x, y])
        road = self._get_road(road_id)
        longitudes = road.longitudes

        # find the closest point of the road to (x,y)
        points = road.points[:, 0:2]
        distance_to_road_points = np.linalg.norm(np.array(points) - point, axis=0)
        closest_point_idx = np.argmin(distance_to_road_points)

        # the point (x,y) should be projected either onto the segment before the closest point or onto the one after it.
        segments_point_idx_pairs = np.array([[closest_point_idx - 1, closest_point_idx],
                                             [closest_point_idx, closest_point_idx + 1]])
        # filter out non-existing indices
        relevant_ind_pairs = segments_point_idx_pairs[np.greater_equal(segments_point_idx_pairs[:, 0], 0) &
                                                      np.less(segments_point_idx_pairs[:, 1], len(points))]

        # for relevant segments, compute (each row): [longitudinal distance of projection on segment,
        # signed lateral distance to the line extending the segment]
        segments_lon_lat = []
        back_of_segment = front_of_segment = 0  # this is used to catch a special case after the for loop
        for segment_idx_pair in relevant_ind_pairs:
            try:
                seg_start = points[segment_idx_pair[0]]
                seg_end = points[segment_idx_pair[1]]

                lon_dist_on_segment = np.linalg.norm(Euclidean.project_on_segment_2d(point, seg_start, seg_end) - seg_start)
                lat_dist_from_seg_extended_line = Euclidean.signed_dist_to_line_2d(point, seg_start, seg_end)

                segments_lon_lat.append([lon_dist_on_segment, lat_dist_from_seg_extended_line])
            except OutOfSegmentBack:
                back_of_segment += 1
                pass
            except OutOfSegmentFront:
                front_of_segment += 1
                pass

        # special case where the point is in the funnel that is created by the normals of two segments
        # at their intersection point. Once this happens, both OutOfSegmentBack and OutOfSegmentFront are raised
        if len(relevant_ind_pairs) == 2 and front_of_segment == 1 and back_of_segment == 1:
            second_seg_start_point_idx = relevant_ind_pairs[1][0]
            distance_to_second_seg_start = np.linalg.norm(point - points[second_seg_start_point_idx])
            segments_lon_lat = [second_seg_start_point_idx, 0.0, distance_to_second_seg_start]

        segments_lon_lat = np.array(segments_lon_lat)

        try:
            # find closest segment by min latitudinal distance
            full_dist_from_segment = np.linalg.norm(segments_lon_lat[:, 0:2], axis=1)
            closest_segment_idx = np.argmin(full_dist_from_segment)
            lon = segments_lon_lat[closest_segment_idx, 0]
            signed_latitude = segments_lon_lat[closest_segment_idx, 1]
            seg_start_point_idx = relevant_ind_pairs[closest_segment_idx][0]

            # longitude (segment offset + longitude on segment), latitude (relative to right side),
            return longitudes[seg_start_point_idx] + lon, signed_latitude + road.width / 2
        except IndexError:  # happens when <segments> is empty
            raise LongitudeOutOfRoad("Tried to project point {} onto road #{} but projection falls outside "
                                     "the road (longitudinally)")

    @raises(RoadNotFound)
    def _shift_road_points_to_latitude(self, road_id, latitude):
        # type: (int, float) -> np.ndarray
        """
        Returns Road.points shifted by <latitude_shift> relative to road's right-side
        :param road_id: road ID to get the points of.
        :param latitude: desired latitude relative to road's right-side
        :return:
        """
        road = self._get_road(road_id)
        return self._shift_road_points(road.points, latitude - road.width / 2)

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
