import copy
from logging import Logger
from typing import List, Union

import numpy as np

from decision_making.src.messages.dds_nontyped_message import DDSNonTypedMsg
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from mapping.src.model.map_api import MapAPI


class RoadLocalization(DDSNonTypedMsg):
    def __init__(self, road_id, lane_num, full_lat, intra_lane_lat, road_lon, intra_lane_yaw):
        # type: (int, int, float, float, float, float, float, float) -> None
        """
        absolute location in road coordinates (road_id, lat, lon)
        :param road_id:
        :param lane_num: 0 is the rightmost
        :param full_lat: in meters; full latitude from the right edge of the road
        :param intra_lane_lat: in meters, 0 is lane left edge
        :param road_lon: in meters, longitude relatively to the road start
        :param intra_lane_yaw: 0 is along road's local tangent
        """
        self.road_id = road_id
        self.lane_num = lane_num
        self.full_lat = full_lat
        self.intra_lane_lat = intra_lane_lat
        self.road_lon = road_lon
        self.intra_lane_yaw = intra_lane_yaw


class RelativeRoadLocalization(DDSNonTypedMsg):
    def __init__(self, rel_lat, rel_lon, rel_yaw):
        # type: (float, float, float) -> None
        """
        relative to ego location in road coordinates (road_id, lat, lon)
        :param rel_lat: in meters, latitude relatively to ego
        :param rel_lon: in meters, longitude relatively to ego
        :param rel_yaw: in radians, yaw relatively to ego
        """
        self.rel_lat = rel_lat
        self.rel_lon = rel_lon
        self.rel_yaw = rel_yaw


class OccupancyState(DDSNonTypedMsg):
    def __init__(self, timestamp, free_space, confidence):
        # type: (int, np.ndarray, np.ndarray) -> None
        """
        free space description
        :param timestamp of free space
        :param free_space: array of directed segments defines a free space border
        :param confidence: array per segment
        """
        self.timestamp = timestamp
        self.free_space = np.copy(free_space)
        self.confidence = np.copy(confidence)


class ObjectSize(DDSNonTypedMsg):
    def __init__(self, length, width, height):
        # type: (float, float, float) -> None
        self.length = length
        self.width = width
        self.height = height


class DynamicObject(DDSNonTypedMsg):
    def __init__(self, obj_id, timestamp, x, y, z, yaw, size, confidence, v_x, v_y, acceleration_lon, omega_yaw,
                 road_localization):
        # type: (int, int, float, float, float, float, ObjectSize, float, float, float, float, float, RoadLocalization) -> None
        """
        both ego and other dynamic objects
        :param obj_id: object id
        :param timestamp: time of perception
        :param x: for ego in world coordinates, for the rest relatively to ego
        :param y:
        :param z:
        :param yaw: for ego 0 means along X axis, for the rest 0 means forward direction relatively to ego
        :param size: class ObjectSize
        :param confidence: of object's existence
        :param v_x: in m/sec; for ego in world coordinates, for the rest relatively to ego
        :param v_y: in m/sec
        :param acceleration_lon: acceleration in longitude axis
        :param omega_yaw: 0 for straight motion, positive for CCW (yaw increases), negative for CW
        """
        self.obj_id = obj_id
        self.timestamp = timestamp
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw
        self.size = copy.copy(size)
        self.confidence = confidence
        self.v_x = v_x
        self.v_y = v_y
        self.road_localization = road_localization
        self.acceleration_lon = acceleration_lon
        self.omega_yaw = omega_yaw

    @property
    def timestamp_in_sec(self):
        return self.timestamp * 1e-9

    @timestamp_in_sec.setter
    def timestamp_in_sec(self, value):
        self.timestamp = int(value * 1e9)


    @property
    def road_longitudinal_speed(self) -> float:
        """
        :return: Longitudinal speed (relative to road)
        """
        return np.linalg.norm([self.v_x, self.v_y]) * np.cos(self.road_localization.intra_lane_yaw)

    @property
    def road_lateral_speed(self) -> float:
        """
        :return: Longitudinal speed (relative to road)
        """
        return np.linalg.norm([self.v_x, self.v_y]) * np.sin(self.road_localization.intra_lane_yaw)

    @staticmethod
    def compute_road_localization(global_pos: np.ndarray, global_yaw: float, map_api: MapAPI) -> RoadLocalization:
        """
        calculate road coordinates for global coordinates for ego
        :param global_pos: 1D numpy array of ego vehicle's [x,y,z] in global coordinate-frame
        :param global_yaw: in global coordinate-frame
        :param map_api: MapAPI instance
        :return: the road localization
        """
        closest_road_id, lon, lat, global_yaw, is_on_road = map_api.convert_global_to_road_coordinates(global_pos[0],
                                                                                                       global_pos[1],
                                                                                                       global_yaw)
        lane_width = map_api.get_road(closest_road_id).lane_width
        lane = np.math.floor(lat / lane_width)
        intra_lane_lat = lat - lane * lane_width

        return RoadLocalization(closest_road_id, int(lane), lat, intra_lane_lat, lon, global_yaw)

    def get_relative_road_localization(self, ego_road_localization, ego_nav_plan, map_api, logger):
        # type: (RoadLocalization, NavigationPlanMsg, MapAPI, Logger) -> Union[RelativeRoadLocalization, None]
        """
        Returns a relative road localization (to given ego state)
        :param logger: logger for debug purposes
        :param ego_road_localization: base road location
        :param ego_nav_plan: the ego vehicle navigation plan
        :param map_api: the map which will be used to calculate the road localization
        :return: a RelativeRoadLocalization object
        """
        relative_lon = map_api.get_longitudinal_difference(initial_road_id=ego_road_localization.road_id,
                                                           initial_lon=ego_road_localization.road_lon,
                                                           final_road_id=self.road_localization.road_id,
                                                           final_lon=self.road_localization.road_lon,
                                                           navigation_plan=ego_nav_plan)

        if relative_lon is None:
            logger.debug("get_point_relative_longitude returned None at DynamicObject.get_relative_road_localization "
                         "for object " + str(self.__dict__))
            return None
        else:
            relative_lat = self.road_localization.full_lat - ego_road_localization.full_lat
            relative_yaw = self.road_localization.intra_lane_yaw - ego_road_localization.intra_lane_yaw
            return RelativeRoadLocalization(rel_lat=relative_lat, rel_lon=relative_lon, rel_yaw=relative_yaw)

    @staticmethod
    def compute_road_localization(global_pos: np.ndarray, global_yaw: float, map_api: MapAPI) -> RoadLocalization:
        """
        calculate road coordinates for global coordinates for ego
        :param global_pos: 1D numpy array of ego vehicle's [x,y,z] in global coordinate-frame
        :param global_yaw: in global coordinate-frame
        :param map_api: MapAPI instance
        :return: the road localization
        """
        closest_road_id, lon, lat, global_yaw, is_on_road = map_api.convert_global_to_road_coordinates(global_pos[0],
                                                                                                       global_pos[1],
                                                                                                       global_yaw)
        lane_width = map_api.get_road(closest_road_id).lane_width
        lane = np.math.floor(lat / lane_width)
        intra_lane_lat = lat - lane * lane_width

        return RoadLocalization(closest_road_id, int(lane), lat, intra_lane_lat, lon, global_yaw)



class EgoState(DynamicObject, DDSNonTypedMsg):
    def __init__(self, obj_id, timestamp, x, y, z, yaw, size, confidence,
                 v_x, v_y, acceleration_lon, omega_yaw, steering_angle, road_localization):
        # type: (int, int, float, float, float, float, ObjectSize, float, float, float, float, float, float, RoadLocalization) -> None
        """
        :param obj_id:
        :param timestamp:
        :param x:
        :param y:
        :param z:
        :param yaw:
        :param size:
        :param confidence:
        :param v_x: in m/sec
        :param v_y: in m/sec
        :param acceleration_lon: in m/s^2
        :param omega_yaw: radius of turning of the ego
        :param steering_angle: equivalent to knowing of turn_radius
        """
        DynamicObject.__init__(self, obj_id, timestamp, x, y, z, yaw, size, confidence, v_x, v_y,
                               acceleration_lon, omega_yaw, road_localization)
        self.steering_angle = steering_angle


class State(DDSNonTypedMsg):
    def __init__(self, occupancy_state, dynamic_objects, ego_state):
        # type: (OccupancyState, List[DynamicObject], EgoState) -> None
        """
        main class for the world state
        :param occupancy_state: free space
        :param dynamic_objects:
        :param ego_state:
        """
        self.occupancy_state = copy.deepcopy(occupancy_state)
        self.dynamic_objects = copy.deepcopy(dynamic_objects)
        self.ego_state = copy.deepcopy(ego_state)
