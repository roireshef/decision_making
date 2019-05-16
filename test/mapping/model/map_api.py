import copy
from abc import ABCMeta
from logging import Logger
from typing import List, Tuple
from typing import Optional

import numpy as np
import six

from decision_making.src.planning.types import FP_SX, FP_DX
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.test.mapping.exceptions import raises, RoadNotFound, MapCellNotFound, LongitudeOutOfRoad, \
    OutOfSegmentBack, OutOfSegmentFront, LaneNotFound
from decision_making.test.mapping.model.constants import DEFAULT_MAP_LAYER
from decision_making.test.mapping.model.localization import RoadCoordinatesDifference, RoadLocalization
from decision_making.test.mapping.model.map_model import MapModel, RoadDetails
from decision_making.test.mapping.transformations.geometry_utils import CartesianFrame, Euclidean
from rte.ctm.src import CtmService

'''
A naive Road-Level MapAPI
'''


@six.add_metaclass(ABCMeta)
class MapAPI:
    def __init__(self, map_model, logger):
        # type: (MapModel, Logger) -> None
        self.logger = logger
        self._cached_map_model = map_model
        self._transform = CtmService.get_ctm()

        self._roads_frenet = {id: FrenetSerret2DFrame.fit(self.__get_road(id)._points)
                              for id in map_model.get_road_ids()}
        self._rhs_roads_frenet = {id: FrenetSerret2DFrame.fit(self._shift_road_points_to_latitude(id, 0))
                              for id in map_model.get_road_ids()}

        # create dictionary of lanes' addresses: lane_id -> (road_segment_id, lane_ordinal)
        # suppose lane_id = road_segment_id * 10 + lane_index
        self._lane_address = {road_segment_id * 10 + lane_ordinal: (road_segment_id, lane_ordinal)
                              for road_segment_id in map_model.get_road_ids()
                              for lane_ordinal in range(map_model.get_road_data(road_segment_id).lanes_num)}

        self._lane_by_address = {lane_addr: lane_id for lane_id, lane_addr in self._lane_address.items()}

        self._lane_points = {lane_id: self._create_center_lane_points(lane_addr[0], lane_addr[1])
                             for lane_id, lane_addr in self._lane_address.items()}

        self._longitudes = {lane_id: RoadDetails.calc_longitudes(self._lane_points[lane_id])
                            for lane_id, _ in self._lane_address.items()}

        self._lane_frenet = {lane_id: FrenetSerret2DFrame.fit(self._lane_points[lane_id])
                             for lane_id, _ in self._lane_address.items()}

        if map_model is not None and map_model._frame_origin is not None:
            # Update the CTM map frame relative to the world frame
            self._transform.update_map_origin(
                latitude=map_model._frame_origin[0],
                longitude=map_model._frame_origin[1]
            )
        else:
            # Happens in a unit test (DM)
            pass

    def update_perceived_roads(self):
        pass

    '''####################'''
    ''' EXTERNAL FUNCTIONS '''
    '''####################'''

    # TODO: document layer argument
    def convert_geo_to_map_coordinates(self, lat, lon):
        # type: (float, float, int) -> [float, float]
        """
        Converts the Geo lat/lon coordinates into map coordinates (meters from the frame origin)
        :param lat: The latitude in degrees
        :param lon: The longitude in degrees
        :param layer:
        :return: The (x,y) position on the map, relative to the frame origin
        """
        return self._transform.transform_geo_location_to_map(lat, lon)

    @raises(RoadNotFound)
    def get_road(self, road_segment_id):
        # type: (int) -> RoadDetails
        return copy.deepcopy(self.__get_road(road_segment_id))

    @raises(RoadNotFound)
    def get_num_lanes(self, road_segment_id):
        # type: (int) -> int
        return self.__get_road(road_segment_id).lanes_num

    @raises(MapCellNotFound, RoadNotFound, LongitudeOutOfRoad)
    def convert_global_to_road_coordinates(self, x, y, yaw):
        # type: (float, float) -> (int, float, float, bool)
        """
        Convert a point in global coordinate frame to road coordinate, by searching for the nearest road and
        projecting it onto this road
        :param x: x coordinate in global coordinate frame
        :param y: y coordinate in global coordinate frame
        :param yaw: the object yaw in in world coordinate in [rad]
        :return: Road ID, longitude from the road's start [m], latitude [m]**relative to road's right-side**,
            is object within road latitudes; intra-road yaw in [rad]
        """
        relevant_road_ids = self._find_roads_containing_point(x, y)
        closest_road_id = self._find_closest_road(x, y, relevant_road_ids)

        lon, lat, yaw = self._convert_global_to_road_coordinates(x, y, yaw, closest_road_id)

        road_width = self.__get_road(closest_road_id).road_width
        is_on_road = bool(0.0 <= lat <= road_width)

        return closest_road_id, lon, lat, yaw, is_on_road

    @raises(RoadNotFound)
    def get_center_lanes_latitudes(self, road_segment_id):
        # type: (int) -> np.ndarray
        """
        Get list of latitudes of all centers of lanes in the road
        :param road_segment_id: Road ID to iterate over its lanes
        :return: numpy array of length as the number of lanes, containing the
                 latitudes [m] of all centers of lanes in the road relative to the right side of the road
        """
        road_details = self.__get_road(road_segment_id)
        lanes_num = road_details.lanes_num
        lane_width = road_details.lane_width
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
            roads_len = [self.__get_road(rid).length for rid in roads_ids]
            return np.add(np.sum(roads_len), -initial_lon + final_lon)
        # look back
        else:
            roads_ids = navigation_plan.road_ids[final_road_idx:initial_road_idx]  # this excludes last road
            roads_len = [self.__get_road(rid).length for rid in roads_ids]
            return -1 * np.add(np.sum(roads_len), - final_lon + initial_lon)

    @raises(RoadNotFound, LongitudeOutOfRoad)
    def advance_on_plan(self, initial_road_id, initial_lon, lookahead_dist, navigation_plan):
        # type: (int, float, float, NavigationPlanMsg) -> (int, float)
        """
        Given a longitude on specific road (<initial_road_id> and <initial_lon>), advance (lookahead) <desired_lon>
        distance. The lookahead iterates over the next roads specified in the <navigation_plan> and returns: (the final
        road id, the longitude along this road). If <desired_lon> is more than the distance to end of the plan, a
        LongitudeOutOfRoad exception is thrown.
        :param initial_road_id: the initial road_segment_id (the vehicle is current on)
        :param initial_lon: initial longitude along <initial_road_id>
        :param lookahead_dist: the desired distance of lookahead in [m].
        :param navigation_plan: the relevant navigation plan to iterate over its road IDs.
        :return: (road_segment_id, longitudinal distance [m] from the beginning of <road_segment_id>)
        """
        current_road_idx_in_plan = navigation_plan.get_road_index_in_plan(initial_road_id)
        roads_ids = navigation_plan.road_ids[current_road_idx_in_plan:]
        roads_len = [self.__get_road(rid).length for rid in roads_ids]

        # distance to roads-ends
        roads_dist_to_end = np.cumsum(np.append([roads_len[0] - initial_lon], roads_len[1:]))
        # how much of lookahead_dist is left after this road
        roads_leftovers = np.subtract(lookahead_dist, roads_dist_to_end)

        try:
            target_road_idx = np.where(roads_leftovers < 0)[0][0]
            return roads_ids[target_road_idx], roads_leftovers[target_road_idx] + roads_len[target_road_idx]
        except IndexError:
            raise LongitudeOutOfRoad("The specified navigation plan is short {} meters to advance {} in longitude"
                                     .format(roads_leftovers[-1], lookahead_dist))

    @raises(RoadNotFound)
    def advance_to_end_of_plan(self, initial_road_id, initial_lon, navigation_plan):
        # type: (int, float, NavigationPlanMsg) -> (int, float, float)
        """
        Given a longitude on specific road (<initial_road_id> and <initial_lon>), advance (lookahead) to the final point
        in the navigation plan.
        :param initial_road_id: the initial road_segment_id (the vehicle is current on)
        :param initial_lon: initial longitude along <initial_road_id>
        :param navigation_plan: the relevant navigation plan to iterate over its road IDs.
        :return: (the last road id, its length [m], total distance to its end point [m])
        """
        initial_road_idx_in_plan = navigation_plan.get_road_index_in_plan(initial_road_id)
        roads_ids = navigation_plan.road_ids[initial_road_idx_in_plan:]
        roads_len = [self.__get_road(rid).length for rid in roads_ids]
        roads_dist_to_end = np.sum(np.append([roads_len[0] - initial_lon], roads_len[1:]))  # dist to roads-ends
        return roads_ids[-1], roads_len[-1], roads_dist_to_end

    @raises(RoadNotFound, LongitudeOutOfRoad)
    def get_lookahead_points(self, initial_road_id, initial_lon, lookahead_dist, desired_lat, navigation_plan):
        # type: (int, float, float, float, NavigationPlanMsg) -> (np.ndarray, float)
        """
        Given a longitude on specific road, return all the points along this (and next) road(s) until reaching
        a lookahead of exactly <desired_lon> meters ahead. In addition, shift all points <desired_lat_shift> laterally,
        relative to the roads right-side.
        :param initial_road_id: the initial road_segment_id (the vehicle is current on)
        :param initial_lon: initial longitude along <initial_road_id>
        :param lookahead_dist: the desired distance of lookahead in [m].
        :param desired_lat: desired lateral shift of points **relative to road's right-side**
        :param navigation_plan: the relevant navigation plan to iterate over its road IDs.
        :return: a numpy array of points size Nx2, and the yaw of initial road longitude [rad]
        """
        # find the final point's (according to desired lookahead distance) road_segment_id and longitude along this road
        final_road_id, final_lon = self.advance_on_plan(initial_road_id, initial_lon, lookahead_dist,
                                                        navigation_plan)
        initial_road_idx = navigation_plan.get_road_index_in_plan(initial_road_id)
        final_road_idx = navigation_plan.get_road_index_in_plan(final_road_id)
        relevant_road_ids = navigation_plan.road_ids[initial_road_idx:(final_road_idx + 1)]

        # exact projection of the initial point and final point on the road
        initial_point = self.convert_road_to_global_coordinates(initial_road_id, initial_lon, desired_lat)
        initial_yaw = initial_point[1]
        initial_pos = initial_point[0][0:2]
        final_pos = self.convert_road_to_global_coordinates(final_road_id, final_lon, desired_lat)[0][:2]

        # shift points (laterally) and concatenate all points of all relevant roads
        shifted_points = np.concatenate([self._shift_road_points_to_latitude(rid, desired_lat)
                                         for rid in relevant_road_ids])

        # calculate accumulate longitudinal distance for all points
        longitudes = np.cumsum(np.concatenate([np.append([0], np.diff(self.__get_road(rid)._longitudes))
                                               for rid in relevant_road_ids]))

        # trim shifted points from both sides according to initial point and final (desired) point
        shifted_points = shifted_points[np.greater(longitudes - initial_lon, 0) &
                                        np.less(longitudes - initial_lon, lookahead_dist)]

        # Build path
        path = np.concatenate(([initial_pos], shifted_points, [final_pos]))

        # Remove duplicate points (start of next road == end of last road)
        path = path[np.append(np.sum(np.diff(path, axis=0), axis=1) != 0.0, [True])]

        return path, initial_yaw

    def get_uniform_path_lookahead(self, road_segment_id, lat_shift, starting_lon, lon_step, steps_num, navigation_plan):
        # type: (int, float, float, float, int, NavigationPlanMsg) -> np.ndarray
        """
        Create array of uniformly distributed points along a given road, shifted laterally by by lat_shift.
        When some road finishes, it automatically continues to the next road, according to the navigation plan.
        The distance between consecutive points is lon_step.
        :param road_segment_id: starting road_segment_id
        :param lat_shift: lateral shift from right side of the road [m]
        :param starting_lon: starting longitude [m]
        :param lon_step: distance between consecutive points [m]
        :param steps_num: output points number
        :param navigation_plan: the relevant navigation plan to iterate over its road IDs.
        :return: uniform sampled points array (Nx2)
        """
        shifted, _ = self.get_lookahead_points(road_segment_id, starting_lon, lon_step * steps_num, lat_shift, navigation_plan)
        # TODO change to precise resampling
        _, resampled, _ = CartesianFrame.resample_curve(curve=shifted, step_size=lon_step)
        return resampled

    @raises(RoadNotFound, LongitudeOutOfRoad)
    def convert_road_to_global_coordinates(self, road_segment_id, lon, lat):
        # type: (int, float, float) -> (np.array, float)
        """
        Given road ID, longitude and latitude along it (relative to the right side of the road), find the matching point
        in global (cartesian) coordinate frame.
        :param road_segment_id: road ID as in the map model
        :param lon: longitude from the beginning of the current road [m]
        :param lat: latitude relative to right side of the road [m]
        :return: numpy array of 3D point [x, y, z] in global coordinate frame, yaw [rad] in global coordinate frame
        """
        frenet = self._rhs_roads_frenet[road_segment_id]
        cpoint = frenet.fpoint_to_cpoint(np.array([lon, lat]))
        global_orientation = frenet.get_yaw(np.array([lon]))[0]

        return np.array([cpoint[0], cpoint[1], 0]), global_orientation

    @raises(RoadNotFound, LaneNotFound, LongitudeOutOfRoad)
    def get_lane_width(self, road_segment_id, lane_num, road_lon):
        # type: (int, int, float) -> (float)
        """
        Returns the road with given id's lane's width in given longitude
        Errors are raised if road or lane are unknown, or longitude is invalid
        :param road_segment_id: The required road id
        :param lane_num: The lane number
        :param road_lon: The longitude on the road where to calculate the lane width
        :return: The lane width at the required road, lane and longitude
        """
        road = self.__get_road(road_segment_id)

        if lane_num < 0 or lane_num >= road._lanes_num:
            raise LaneNotFound('Unknown road with id %s' % road_segment_id)
        if road_lon < 0 or road_lon > road.length:
            raise LongitudeOutOfRoad('Longitude %s is out of road [%s,%s]' % (road_lon, 0, road.length))

        return road._lane_width

    def compute_road_localization(self, global_pos, global_yaw):
        # type: (np.ndarray, float) -> RoadLocalization
        """
        calculate road coordinates for global coordinates for ego
        :param global_pos: 1D numpy array of ego vehicle's [x,y,z] in global coordinate-frame
        :param global_yaw: in global coordinate-frame
        :return: the road localization
        """
        closest_road_id, lon, lat, intra_road_yaw, is_on_road = \
            self.convert_global_to_road_coordinates(global_pos[0], global_pos[1], global_yaw)
        lane_width = self.get_road(closest_road_id).lane_width
        lane = np.math.floor(lat / lane_width)
        intra_lane_lat = lat - lane * lane_width

        return RoadLocalization(closest_road_id, int(lane), lat, intra_lane_lat, lon, intra_road_yaw)

    def compute_road_localizations_diff(self, reference_localization, object_localization, navigation_plan):
        # type: (RoadLocalization, RoadLocalization, NavigationPlanMsg) -> Optional[RoadCoordinatesDifference]
        """
        Returns a relative road localization (to given reference object)
        :param reference_localization: reference object's road location
        :param object_localization: object's road location
        :param navigation_plan: the ego vehicle navigation plan
        :return: a RelativeRoadLocalization object
        """
        relative_lon = self.get_longitudinal_difference(initial_road_id=reference_localization.road_id,
                                                        initial_lon=reference_localization.road_lon,
                                                        final_road_id=object_localization.road_id,
                                                        final_lon=object_localization.road_lon,
                                                        navigation_plan=navigation_plan)

        if relative_lon is None:
            self.logger.debug("get_point_relative_longitude returned None at MapApi.get_relative_road_localization "
                              "for object %s and reference %s", object_localization, reference_localization)
            return None
        else:
            relative_lat = object_localization.intra_road_lat - reference_localization.intra_road_lat
            relative_yaw = object_localization.intra_road_yaw - reference_localization.intra_road_yaw
            relative_lane = object_localization.lane_num - reference_localization.lane_num
            return RoadCoordinatesDifference(rel_lat=relative_lat, rel_lon=relative_lon, rel_yaw=relative_yaw,
                                             rel_lane=relative_lane)

    '''####################'''
    ''' INTERNAL FUNCTIONS '''
    '''####################'''

    @raises(RoadNotFound)
    def __get_road(self, road_segment_id):
        # type: (int) -> RoadDetails
        return self._cached_map_model.get_road_data(road_segment_id)

    @raises(MapCellNotFound)
    def _find_roads_containing_point(self, x, y):
        # type: (float, float) -> List[int]
        """
        Returns the list of corresponding road IDs to a coordinate in the global-frame (x, y)
        :param x: world coordinates [m]
        :param y: world coordinates [m]
        :return: a list of road_ids containing the point (x, y)
        """
        # TODO: unify cell-from-xy computation with the one in the map's creation procedure,
        tile_size = self._cached_map_model.xy2road_tile_size
        cell_x = int(round(x / tile_size))
        cell_y = int(round(y / tile_size))
        return self._cached_map_model.get_xy2road_cell((DEFAULT_MAP_LAYER, cell_x, cell_y))

    @raises(RoadNotFound)
    def _find_closest_road(self, x, y, road_segment_ids):
        # type: (float, float, List[int]) -> int
        """
        Returns the closest road_segment_id of the road which is closest to a point in the world (x, y).
        :param x: x coordinate on map (given in [m])
        :param y: y coordinate on map (given in [m])
        :param road_segment_ids: list of road IDs to try to project the point on
        :return: road_segment_id of the closest road
        """
        distances = [self._dist_to_road(x, y, rid) for rid in road_segment_ids]
        return road_segment_ids[np.argmin(distances)]

    @raises(RoadNotFound)
    def _dist_to_road(self, x, y, road_segment_id):
        # type: (float, float, int) -> float
        """
        Compute distance to road by looking for its nearest point to (x,y), and computing the distance-to-segment to
        the segment before and after the nearest point
        :param x: x coordinate on map (given in [m])
        :param y: y coordinate on map (given in [m])
        :param road_segment_id: the road id to measure distance to
        :return: road ID to compute the distance to
        """
        point = np.array([x, y])
        road = self.__get_road(road_segment_id)
        points = road._points[:, 0:2]

        # Get closest segments to point (x,y)
        closest_point_idx_pairs = Euclidean.get_indexes_of_closest_segments_to_point(point, points)

        # Get distance from (x,y) to those segments
        segments_dists = [Euclidean.dist_to_segment_2d(point, points[pair[0]], points[pair[1]])
                          for pair in closest_point_idx_pairs]

        return min(segments_dists)

    @raises(RoadNotFound, LongitudeOutOfRoad)
    def _convert_global_to_road_coordinates(self, x, y, yaw, road_segment_id):
        # type: (float, float, int) -> (float, float, float)
        """
        Convert point in world coordinates (x, y) to (lat, lon) of road with given road_segment_id
        If the point is on the road (in the sense of longitude), then lat is also the distance between the point
        and the road. Otherwise lat is the distance but not latitude, because in this case latitude is meaningless.
        :param x: the point's world x coordinate in meters
        :param y: the point's world y coordinate in meters
        :param yaw: the object yaw in in world coordinate in [rad]
        :param road_segment_id: road ID as in the map model
        :return: longitude (from road start), latitude **relative to road's right side**, intra-road yaw [rad]
        """
        point = np.array([x, y])
        try:
            frenet = self._rhs_roads_frenet[road_segment_id]
            fpoint = frenet.cpoint_to_fpoint(point)
            relative_yaw = yaw - frenet.get_yaw(fpoint[FP_SX])
        except OutOfSegmentBack or OutOfSegmentFront:
            raise LongitudeOutOfRoad("LongitudeOutOfRoad: Tried to project point %s onto road #%s but projection "
                                     "falls outside the road (longitudinally)" % (point, road_segment_id))

        return fpoint[FP_SX], fpoint[FP_DX], relative_yaw

    @raises(RoadNotFound)
    def _shift_road_points_to_latitude(self, road_segment_id, latitude):
        # type: (int, float) -> np.ndarray
        """
        Returns Road.points shifted to the given latitude relative to road's right-side
        :param road_segment_id: road ID to get the points of.
        :param latitude: desired latitude relative to road's right-side [m]
        :return: numpy array (Nx2) of points with the given latitude
        """
        road = self.__get_road(road_segment_id)

        # Uses the fact that the latitude of the road points is center road, i.e., road_width/2, in order to shift them
        # to the given latitude
        return self._shift_road_points(road._points, latitude - road.road_width / 2)

    @staticmethod
    def _shift_road_points(points, lateral_shift):
        # type: (np.ndarray, float) -> np.ndarray
        """
        Given points list along a road, shift them laterally by lat_shift [m]
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

    def _create_center_lane_points(self, road_segment_id, lane_ordinal):
        # type: (int, int) -> np.ndarray
        lane_width = self.__get_road(road_segment_id).lane_width
        latitude = lane_width * (lane_ordinal + 0.5)
        return self._shift_road_points_to_latitude(road_segment_id, latitude)

    @staticmethod
    def _normalize_matrix_rows(mat):
        # type: (np.array) -> np.array
        """
        normalize vector, prevent division by zero
        :param mat: 2D numpy array
        :return: normalized vector (numpy array) with the same dimensions as mat
        """
        norms = np.linalg.norm(mat, axis=1)[np.newaxis].T
        norms[np.where(norms == 0.0)] = 1.0
        return np.divide(mat, norms)

    def get_road_center_frenet_frame(self, road_segment_id: int) -> FrenetSerret2DFrame:
        """
        Get cached Frenet frame of the road center
        :param road_segment_id: road segment id
        :return: cached Frenet frame
        """
        # Get Object's Frenet frame
        return self._roads_frenet[road_segment_id]

    def get_frame_origin(self):
        """

        :return:
        """
        return self._cached_map_model._frame_origin
