import copy
from typing import List

import numpy as np
from decision_making.src.map.constants import *
from decision_making.src.map.map_api import MapAPI

from decision_making.src.messages.dds_typed_message import DDSTypedMsg


class RoadLocalization(DDSTypedMsg):
    def __init__(self, road_id, lane_num, intra_lane_lat, road_lon, intra_lane_yaw,
                 lon_confidence, lat_confidence):
        # type: (int, int, float, float, float, float, float) -> None
        """
        location in road coordinates (road_id, lat, lon)
        :param road_id:
        :param lane_num: 0 is the leftmost
        :param intra_lane_lat: in meters, 0 is lane left edge
        :param road_lon: in meters, longitude relatively to the road start
        :param intra_lane_yaw: 0 is along road's local tangent
        :param lon_confidence: confidence of road_id & road_lon
        :param lat_confidence: confidence of lane, intra_lane params
        """
        self.road_id = road_id
        self.lane_num = lane_num
        self.intra_lane_lat = intra_lane_lat
        self.road_lon = road_lon
        self.intra_lane_yaw = intra_lane_yaw
        self.lon_confidence = lon_confidence
        self.lat_confidence = lat_confidence


class RelativeRoadLocalization(DDSTypedMsg):
    def __init__(self, rel_lat, rel_lon, rel_yaw):
        # type: (float, float, float) -> None
        """
        location in road coordinates (road_id, lat, lon)
        :param rel_lat: in meters, latitude relatively to ego
        :param rel_lon: in meters, longitude relatively to ego
        :param rel_yaw: in radians, yaw relatively to ego
        """
        self.rel_lat = rel_lat
        self.rel_lon = rel_lon
        self.rel_yaw = rel_yaw


class OccupancyState(DDSTypedMsg):
    def __init__(self, timestamp, free_space, confidence):
        # type: (int, np.ndarray, np.ndarray) -> None
        """
        free space description
        :param timestamp of free space
        :param free_space: array of directed segments defines a free space border
        :param confidence: array per segment
        """
        self._timestamp = timestamp
        self.free_space = np.copy(free_space)
        self.confidence = np.copy(confidence)


class ObjectSize(DDSTypedMsg):
    def __init__(self, length, width, height):
        # type: (float, float, float) -> None
        self.length = length
        self.width = width
        self.height = height


class DynamicObject(DDSTypedMsg):
    def __init__(self, obj_id, timestamp, x, y, z, yaw, size, road_localization, rel_road_localization,
                 confidence, localization_confidence, v_x, v_y, acceleration_lon, turn_radius):
        # type: (int, int, float, float, float, float, ObjectSize, Union[RoadLocalization, None], Union[RelativeRoadLocalization, None], float, float, float, float, Union[float, None], Union[float, None]) -> None
        """
        both ego and other dynamic objects
        :param obj_id: object id
        :param timestamp: time of perception
        :param x: for ego in world coordinates, for the rest relatively to ego
        :param y:
        :param z:
        :param yaw: for ego 0 means along X axis, for the rest 0 means forward direction relatively to ego
        :param size: class ObjectSize
        :param road_localization: class RoadLocalization
        :param rel_road_localization: class RelativeRoadLocalization (relative to ego)
        :param confidence: of object's existence
        :param localization_confidence: of location
        :param v_x: in m/sec; for ego in world coordinates, for the rest relatively to ego
        :param v_y: in m/sec
        :param acceleration_lon: acceleration in longitude axis
        :param turn_radius: 0 for straight motion, positive for CW (yaw increases), negative for CCW
        """
        self.id = obj_id
        self._timestamp = timestamp
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw
        self.size = copy.copy(size)
        self.confidence = confidence
        self.localization_confidence = localization_confidence
        self.v_x = v_x
        self.v_y = v_y

        if road_localization is not None:
            self.road_localization = copy.copy(road_localization)
        else:
            raise NotImplementedError()

        if rel_road_localization is not None:
            self.rel_road_localization = copy.copy(rel_road_localization)
        else:
            raise NotImplementedError()

        if acceleration_lon is not None:
            self.acceleration_lon = acceleration_lon
        else:
            raise NotImplementedError()

        if turn_radius is not None:
            self.turn_radius = turn_radius
        else:
            raise NotImplementedError()

    def predict(self, goal_timestamp, map_api) -> None:
        # type: (int, MapAPI) -> DynamicObject
        """
        Predict the object's location for the future timestamp
        !!! This function changes the object's location, velocity and timestamp !!!
        :param goal_timestamp: the goal timestamp for prediction
        :param lane_width: closest lane_width
        :return: None
        """
        pass

class EgoState(DynamicObject, DDSTypedMsg):
    def __init__(self, obj_id, timestamp, x, y, z, yaw, size, road_localization, rel_road_localization, confidence,
                 localization_confidence, v_x, v_y, acceleration_lon, turn_radius, steering_angle):
        # type: (int, int, float, float, float, float, ObjectSize, RoadLocalization, Union[RelativeRoadLocalization, None], float, float, float, float, Union[float, None], Union[float, None], Union[float, None]) -> None
        """
        :param obj_id:
        :param timestamp:
        :param x:
        :param y:
        :param z:
        :param yaw:
        :param size:
        :param road_localization:
        :param confidence:
        :param localization_confidence:
        :param v_x: in m/sec
        :param v_y: in m/sec
        :param acceleration_lon: in m/s^2
        :param turn_radius: radius of turning of the ego
        :param steering_angle: equivalent to knowing of turn_radius
        """
        DynamicObject.__init__(self, obj_id, timestamp, x, y, z, yaw, size, road_localization, rel_road_localization,
                               confidence, localization_confidence, v_x, v_y, acceleration_lon, turn_radius)
        if steering_angle is not None:
            self.steering_angle = steering_angle
        else:
            raise NotImplementedError()


class LanesStructure(DDSTypedMsg):
    def __init__(self, center_of_lane_points, width_vec):
        # type: (np.ndarray, np.ndarray) -> None
        """
        this class is instantiated for each lane
        :param center_of_lane_points:  points array for a given lane
        :param width_vec:  array of width: lane width per lane point
        """
        self.center_of_lane_points = copy.deepcopy(center_of_lane_points)
        self.width_vec = copy.deepcopy(width_vec)


class PerceivedRoad(DDSTypedMsg):
    def __init__(self, timestamp, lanes_structure, confidence):
        # type: (int, List[LanesStructure], float) -> None
        """
        the road of ego as it viewed by perception
        :param timestamp:
        :param lanes_structure: list of elements of type LanesStructure, per lane
        :param confidence:
        """
        self.timestamp = timestamp
        self.lanes_structure = copy.deepcopy(lanes_structure)
        self.confidence = confidence

class State(DDSTypedMsg):
    def __init__(self, occupancy_state, dynamic_objects, ego_state, perceived_road):
        # type: (OccupancyState, List[DynamicObject], EgoState, PerceivedRoad) -> None
        """
        main class for the world state
        :param occupancy_state: free space
        :param dynamic_objects:
        :param ego_state:
        :param perceived_road: the road of ego as it viewed by perception, relatively to ego
        """
        self.occupancy_state = copy.deepcopy(occupancy_state)
        self.dynamic_objects = copy.deepcopy(dynamic_objects)
        self.ego_state = copy.deepcopy(ego_state)
        self.perceived_road = copy.deepcopy(perceived_road)

    @classmethod
    def create_empty(cls):
        occupancy_state = OccupancyState(0, np.array([]), np.array([]))
        dynamic_objects = []
        size = ObjectSize(0, 0, 0)
        road_localization = RoadLocalization(0, 0, 0, 0, 0, 0, 0)
        rel_road_localization = RelativeRoadLocalization(0, 0, 0)
        ego_state = EgoState(0, 0, 0, 0, 0, 0, size, road_localization, rel_road_localization, 0, 0, 0, 0, 0, 0, 0)
        perceived_road = PerceivedRoad(0, [], 0)
        state = cls(occupancy_state, dynamic_objects, ego_state, perceived_road)
        return state

    def update_objects(self):
        """
        insert object to state - will be implemented by Ron
        :return: merged State
        """
        pass

    def update_ego_state(self):
        """
        insert ego localization to state - will be implemented by Ron
        :return: merged State
        """
        pass

    def predict(self, goal_timestamp, map_api):
        # type: (int, MapAPI) -> State
        """
        predict the ego localization, other objects and free space for the future timestamp
        :param goal_timestamp:
        :return:
        """
        pass
