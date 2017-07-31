import numpy as np
from .map_model import MapModel
from abc import ABCMeta, abstractmethod

LARGE_NUM = 1000000


class CacheMap(metaclass=ABCMeta):
    def __init__(self):
        self.cached_map_model = MapModel()
        pass

    @abstractmethod
    def load_map(self, map_model_pickle_filename):  # abstract method
        """
        abstract method; is implemented in NaiveCacheMap
        :param map_model_pickle_filename: python "pickle" file containing the dictionaries of map_model
        :return:
        """
        pass

    def get_road_attribute(self, road_id, attribute):
        """
        shortcut to road's attribute
        :param road_id:
        :param attribute: e.g. lanes num, center points
        :return: the road's attribute
        """
        return self.cached_map_model.roads_data[road_id][attribute]

    def calc_distance_between_point_and_road(self, x, y, road_id):
        """
        calc distance between point=(x,y) and a given road. if the point is on the road, return 0
        :param x: the point's world x coordinate in meters
        :param y: the point's world y coordinate in meters
        :param road_id:
        :return: distance between (x,y) and the road
        """
        points = self.get_road_attribute(road_id, 'points').transpose()
        road_width = self.get_road_attribute(road_id, 'width')
        dist_2 = np.linalg.norm(np.asarray(points) - (x, y), axis=1)
        closest_i = np.argmin(dist_2)
        # find the closest segment and the distance (latitude)
        dist1 = dist2 = LARGE_NUM
        if closest_i > 0:
            _, dist1, _ = CalcPointSegmentDist([x, y], points[closest_i - 1], points[closest_i])
        if closest_i < len(points) - 1:
            _, dist2, _ = CalcPointSegmentDist([x, y], points[closest_i], points[closest_i + 1])
        return min(dist1, dist2) - 0.5 * road_width

    def xy2road(self, layer, x, y):
        """
        shortcut to a cell of the map xy2road_map
        :param layer: 0 ground, 1 on bridge, 2 bridge above bridge, etc
        :param x: world coordinates in meters
        :param y: world coordinates in meters
        :return: road_ids containing the point x, y
        """
        X = int(round(x / self.cached_map_model.xy2road_map_cell_size))
        Y = int(round(y / self.cached_map_model.xy2road_map_cell_size))
        if (layer, X, Y) in self.cached_map_model.xy2road_map:
            return self.cached_map_model.xy2road_map[(layer, X, Y)]
        return None

    def get_center_lanes_latitudes(self, road_id):
        """
        get list of latitudes of all lanes in the road
        :param road_id:
        :return: list of latitudes of all lanes in the road
        """
        lanes_num = self.get_road_attribute(road_id, 'lanes')
        road_width = self.get_road_attribute(road_id, 'width')
        lane_width = float(road_width) / lanes_num
        center_lanes = lane_width / 2 + np.array(range(lanes_num)) * lane_width
        return center_lanes

    def get_road_details(self, road_id):
        """
        get details of a given road
        :param road_id:
        :return: lanes number, road width, road length, road's points
        """
        if road_id not in self.cached_map_model.roads_data.keys():
            return None, None, None, None
        lanes_num = self.get_road_attribute(road_id, 'lanes')
        width = self.get_road_attribute(road_id, 'width')
        length = self.get_road_attribute(road_id, 'longitudes')[-1]
        points = np.array(self.get_road_attribute(road_id, 'points'))
        return lanes_num, width, length, points

    def get_point_in_road_coordinates(self, x, y, z, yaw):
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
            road_ids = self.xy2road(1, x, y)
        else:
            road_ids = self.xy2road(0, x, y)
            if road_ids is None:
                road_ids = self.xy2road(1, x, y)
        if road_ids is None or len(road_ids) == 0:
            return None, None, None, None, None, None, None

        closest_dist = LARGE_NUM
        closest_id = closest_sign = closest_yaw = closest_lon = closest_lat = None
        for road_id in road_ids:
            longitudes = self.get_road_attribute(road_id, 'longitudes')
            road_width = self.get_road_attribute(road_id, 'width')
            # find the closest point of the road to (x,y)
            points = self.get_road_attribute(road_id, 'points')[0:2].transpose()
            dist_2 = np.linalg.norm(np.asarray(points) - (x, y), axis=1)
            closest_i = np.argmin(dist_2)
            # find the closest segment and the distance (latitude)
            p = proj1 = proj2 = [x, y]
            lat_dist1 = lat_dist2 = LARGE_NUM
            sign1 = sign2 = 0
            if closest_i > 0:
                sign1, lat_dist1, proj1 = CalcPointSegmentDist(p, points[closest_i - 1], points[closest_i])
            if closest_i < len(points) - 1:
                sign2, lat_dist2, proj2 = CalcPointSegmentDist(p, points[closest_i], points[closest_i + 1])
            if lat_dist1 < lat_dist2:
                lat_dist = lat_dist1
                sign = sign1
                lon = proj1 + longitudes[closest_i - 1]
                road_yaw = np.arctan2(points[closest_i][1] - points[closest_i - 1][1],
                                      points[closest_i][0] - points[closest_i - 1][0])
            else:
                lat_dist = lat_dist2
                sign = sign2
                lon = proj2 + longitudes[closest_i]
                road_yaw = np.arctan2(points[closest_i + 1][1] - points[closest_i][1],
                                      points[closest_i + 1][0] - points[closest_i][0])
            dist = max(0, lat_dist - 0.5 * road_width)
            if dist < closest_dist:
                closest_dist = dist
                closest_lat = lat_dist
                closest_sign = sign
                closest_lon = lon
                closest_yaw = road_yaw
                closest_id = road_id

        road_id = closest_id
        lat_dist = closest_lat
        sign = closest_sign
        lon = closest_lon
        road_yaw = closest_yaw
        lanes_num = self.get_road_attribute(road_id, 'lanes')
        lane_width = self.get_road_attribute(road_id, 'width') / float(lanes_num)

        full_lat = lat_dist * sign + 0.5 * lanes_num * lane_width  # latitude relatively to the left road edge
        lane = float(int(full_lat / lane_width))  # from left to right
        lane = np.clip(lane, 0, lanes_num - 1)
        yaw_in_road = (yaw - road_yaw + 2 * np.pi) % 2 * np.pi
        lane_lat = full_lat % lane_width
        return road_id, lane, full_lat, lane_lat, lon, yaw_in_road

    def get_point_relative_longitude(self, from_road_id, from_lon_in_road, to_road_id, to_lon_in_road,
                                     max_lookahead_distance, navigation_plan):
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
                total_lon_distance = self.get_road_attribute(from_road_id, 'longitudes')[-1] - from_lon_in_road
            else:  # backward
                total_lon_distance = from_lon_in_road

            # 2. Middle road segments
            road_id, road_index_in_plan = navigation_plan.get_next_road(direction, road_index_in_plan)
            while road_id is not None and road_id != to_road_id and total_lon_distance < max_lookahead_distance:
                road_length = self.get_road_attribute(road_id, 'longitudes')[-1]
                total_lon_distance += road_length
                road_id, road_index_in_plan = navigation_plan.get_next_road(direction, road_index_in_plan)

            # 3. Add length of last road segment
            if road_id == to_road_id:
                if direction > 0:  # forward
                    total_lon_distance += to_lon_in_road
                else:  # backward
                    total_lon_distance += self.get_road_attribute(to_road_id, 'longitudes')[-1] - to_lon_in_road
                found_connection = True
                break  # stop the search when the connection is found

        return total_lon_distance, found_connection

    def shift_road_vector_in_lat(self, points, lat_shift):
        """
        Given points list along a road, shift them laterally by lat_shift meters
        :param points: points list along a road
        :param lat_shift: shift in meters
        :return: shifted points array
        """
        points = np.array(points)
        points_direction = np.diff(points, axis=1)
        points_norm = np.linalg.norm(points_direction, axis=0)
        normalized_vec_x = np.divide(points_direction[0, :], points_norm)
        normalized_vec_y = np.divide(points_direction[1, :], points_norm)
        lat_vec = np.vstack((normalized_vec_y, -normalized_vec_x))
        lat_vec = np.concatenate((lat_vec, lat_vec[:, -1].reshape([2, 1])), axis=1)
        shifted_points = points + lat_vec * lat_shift
        return shifted_points

    def get_path_lookahead(self, road_id, lon, lat, max_lookahead_distance, navigation_plan, direction=1):
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

        road_width = self.get_road_attribute(road_id, 'width')
        center_road_lat = road_width / 2.0

        longitudes = np.array(self.get_road_attribute(road_id, 'longitudes'))
        path = np.zeros(shape=[2, 0])

        road_length = self.get_road_attribute(road_id, 'longitudes')[-1]
        closest_longitude_index = np.argmin(np.abs(longitudes - lon))

        # Init with current road
        road_starting_longitude = lon
        residual_lookahead = max_lookahead_distance

        path_points = np.array(self.get_road_attribute(road_id, 'points'))
        partial_path_points = np.zeros((2, 0))
        achieved_lookahead = road_length - lon
        if direction == 1:
            target_longitude_index = np.argmin(np.abs(longitudes - (residual_lookahead + road_starting_longitude)))
            first_exact_lon_point = self.lat_lon_2_world(road_id, center_road_lat, lon, navigation_plan)
            partial_path_points = path_points[:, closest_longitude_index:target_longitude_index + 1]
            partial_path_points[:, 0] = first_exact_lon_point[0:2]
            achieved_lookahead = road_length - lon
        else:
            dummy_do_nothing = True
            # target_longitude_index = np.argmin(np.abs((road_length - longitudes) - residual_lookahead))
            # first_exact_lon_point = self.LatLon2World(road_id=road_id, lon=lon, lat=lat)
            # partial_path_points = path_points[:, target_longitude_index:closest_longitude_index+1]
            # partial_path_points = np.flip(partial_path_points, axis=1)
            # achieved_lookahead = lon

        path = np.concatenate((path, partial_path_points), axis=1)
        # Iterate over next road, until we get enough lookahead
        while achieved_lookahead < max_lookahead_distance and road_id is not None:
            road_starting_longitude = 0
            road_id, road_index_in_plan = navigation_plan.get_next_road(direction, road_index_in_plan)

            if road_id is not None:
                road_length = self.get_road_attribute(road_id, 'longitudes')[-1]
                path_points = np.array(self.get_road_attribute(road_id, 'points'))

                residual_lookahead = max_lookahead_distance - achieved_lookahead
                if road_length > residual_lookahead:
                    # Take only the relevant part of the current road
                    longitudes = np.array(self.get_road_attribute(road_id, 'longitudes'))
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
            dx = path[0, -1] - partial_path_points[0, 0]
            dy = path[1, -1] - partial_path_points[1, 0]
            if dx * dx + dy * dy < 0.0001:  # avoid duplicated start point of the next path
                from_idx = 1
            path = np.concatenate((path, partial_path_points[:, from_idx:]), axis=1)
            achieved_lookahead += road_length

        # Replace the last (closest, but inexact) point, and replace it with a point with the exact lon value
        if direction == 1:
            lon = residual_lookahead + road_starting_longitude
        else:
            lon = road_length - (road_starting_longitude + residual_lookahead)
        last_exact_lon_point = self.lat_lon_2_world(road_id, center_road_lat, lon, navigation_plan)

        # if path consists of a single point, add it to the end of route. else, replace last point
        path_length = path.shape[1]
        if path_length > 1:
            path[:, -1] = last_exact_lon_point[0:2]
        else:
            dx = path[0, -1] - last_exact_lon_point[0]
            dy = path[1, -1] - last_exact_lon_point[1]
            if dx * dx + dy * dy > 0.0001:  # avoid duplicated point of the next path
                path = np.concatenate((path, last_exact_lon_point[0:2].reshape([2, 1])), axis=1)

        if lat != 0:
            shift_amount = -road_width / 2.0 + lat
            lat_shifted_path = self.shift_road_vector_in_lat(points=path, lat_shift=shift_amount)
        else:
            lat_shifted_path = path

        return lat_shifted_path

    def get_uniform_path_lookahead(self, road_id, lat, starting_lon, lon_step, steps_num):
        """
        Create array of uniformly distanced points along the given road, shifted by lat.
        When some road finishes, it automatically continues to the next road, according to the navigation plan.
        The distance between consecutive points is lon_step.
        :param road_id: starting road_id
        :param lat: lateral shift
        :param starting_lon: starting longitude
        :param lon_step: distance between consecutive points
        :param steps_num: output points number
        :return: uniform points array
        """
        pnt_ind = 0
        points = np.zeros(shape=[2, 0])
        lat_vecs = np.zeros(shape=[2, 0])
        lon = starting_lon
        road_index_in_plan = None
        for step in range(steps_num):
            road_id, _, left_point, lat_vec, pnt_ind, lon, road_index_in_plan = \
                self.lon_2_world(road_id, pnt_ind, lon, road_index_in_plan)
            world_pnt = [left_point[0] + lat_vec[0] * lat, left_point[1] + lat_vec[1] * lat]
            points = np.concatenate((points, np.array(world_pnt).reshape([2, 1])), axis=1)
            lat_vecs = np.concatenate((lat_vecs, np.array(lat_vec).reshape([2, 1])), axis=1)
            lon += lon_step
        return points, lat_vecs

    def lon_2_world(self, road_id, pnt_ind, road_lon, navigation_plan, road_index_in_plan=None):
        """
        Calculate world point matching to a given longitude of a given road.
        If the given longitude exceeds the current road length, then calculate the point in the next road.
        The next road is picked from the navigation plan.
        :param road_id: current road_id
        :param pnt_ind: index of the road point, from which we can search the new point (for speed optimization)
        :param road_lon: the point's longitude relatively to the start of the road_id
        :param road_index_in_plan: current index in the navigation plan
        :return: new road_id (maybe the same one) and its length,
            most left road point at the given longitude with latitude vector (perpendicular to the local road's tangent),
            the first point index with longitude > the given longitude (for the next search)
            the longitude relatively to the next road (in case if the road_id has changed)
            the navigation plan index
        """
        if road_index_in_plan is None:
            road_index_in_plan = navigation_plan.get_road_index_in_plan(road_id)

        longitudes = self.get_road_attribute(road_id, 'longitudes')
        if road_lon > longitudes[-1]:  # then advance to the next road
            road_lon -= longitudes[-1]
            road_id, road_index_in_plan = navigation_plan.get_next_road(direction=1, road_index=road_index_in_plan)
            pnt_ind = 1
            longitudes = self.get_road_attribute(road_id, 'longitudes')
        points = self.get_road_attribute(road_id, 'points')[0:2].transpose()
        width = self.get_road_attribute(road_id, 'width')
        length = self.get_road_attribute(road_id, 'longitudes')[-1]
        # get point with longitude > cur_lon
        while pnt_ind < len(points) - 1 and road_lon > longitudes[pnt_ind]:
            pnt_ind += 1
        pnt_ind = max(1, pnt_ind)
        lane_vec = [points[pnt_ind][0] - points[pnt_ind - 1][0], points[pnt_ind][1] - points[pnt_ind - 1][1]]
        lane_vec_len = np.sqrt(lane_vec[0] * lane_vec[0] + lane_vec[1] * lane_vec[1])
        lane_vec = [lane_vec[0] / lane_vec_len, lane_vec[1] / lane_vec_len]
        lat_vec = [lane_vec[1], -lane_vec[0]]
        center_point = [points[pnt_ind][0] - lane_vec[0] * (longitudes[pnt_ind] - road_lon),
                        points[pnt_ind][1] - lane_vec[1] * (longitudes[pnt_ind] - road_lon)]
        left_point = [center_point[0] - lat_vec[0] * width / 2., center_point[1] - lat_vec[1] * width / 2.]
        # if longitudes[pnt_ind] - cur_lon > cur_lon - longitudes[pnt_ind-1]:
        #    pnt_ind -= 1
        return road_id, length, left_point, lat_vec, pnt_ind, road_lon, road_index_in_plan

    def lat_lon_2_world(self, road_id, lat, lon, navigation_plan):
        """
        Given road_id, lat & lon, calculate the point in world coordinates.
        Z coordinate is calculated using the OSM data: if road's head and tail are at different layers (height),
        then interpolate between them.
        :param road_id:
        :param lat:
        :param lon:
        :return: point in 3D world coordinates
        """
        id, length, left_point, lat_vec, _, _, _ = self.lon_2_world(road_id, 0, lon, navigation_plan)
        if id != road_id:
            return None
        head_layer = self.get_road_attribute(road_id, 'head_layer')
        tail_layer = self.get_road_attribute(road_id, 'tail_layer')
        tail_wgt = lon / length
        Z = head_layer * (1 - tail_wgt) + tail_layer * tail_wgt
        world_pnt = np.array([left_point[0] + lat_vec[0] * lat, left_point[1] + lat_vec[1] * lat, Z])
        return world_pnt

    def update_perceived_roads(self):
        pass


def CalcPointSegmentDist(p, p1, p2):
    """
    Given point p and directed segment p1->p2, calculate:
        1. from which side p is located relatively to the line p1->p2,
        2. the closest distance from p to the segment,
        3. length of the projection of p on the segment (zero if the projection is outside the segment).
    :param p: 2D Point
    :param p1: first edge of 2D segment
    :param p2: second edge of 2D segment
    :return: the distance between the point p and the segment p1->p2.
    """
    v = [p2[0] - p1[0], p2[1] - p1[1]]
    v1 = [p[0] - p1[0], p[1] - p1[1]]
    v2 = [p2[0] - p[0], p2[1] - p[1]]
    if v[0] == 0 and v[1] == 0:
        return 0, np.sqrt(v1[0] * v1[0] + v1[1] * v1[1]), 0
    dot1 = v[0] * v1[0] + v[1] * v1[1]
    dot2 = v[0] * v2[0] + v[1] * v2[1]
    normal = [v[1], -v[0]]  # normal of v toward right if v looks up
    dotn = normal[0] * v1[0] + normal[1] * v1[1]
    sign = np.sign(dotn)
    proj = 0
    if dot1 > 0 and dot2 > 0:  # then p is between p1,p2, so calc dist to the line
        one_over_vnorm = 1. / np.sqrt(v[0] * v[0] + v[1] * v[1])
        dist = dotn * one_over_vnorm * sign  # always >= 0
        proj = dot1 * one_over_vnorm  # length of projection of v1 on v
    elif dot1 <= 0:
        dist = np.sqrt(v1[0] * v1[0] + v1[1] * v1[1])
    else:
        dist = np.sqrt(v2[0] * v2[0] + v2[1] * v2[1])
    return sign, dist, proj
