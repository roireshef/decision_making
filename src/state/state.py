import copy
from typing import List, Union

import numpy as np
#from decision_making.src.map.constants import *
from decision_making.src.map.map_api import MapAPI
from decision_making.src.messages.dds_nontyped_message import DDSNonTypedMsg
from decision_making.src.planning.utils.geometry_utils import CartesianFrame


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
        self._timestamp = timestamp
        self.free_space = np.copy(free_space)
        self.confidence = np.copy(confidence)


class ObjectSize(DDSNonTypedMsg):
    def __init__(self, length, width, height):
        # type: (float, float, float) -> None
        self.length = length
        self.width = width
        self.height = height


class DynamicObject(DDSNonTypedMsg):
    def __init__(self, obj_id, timestamp, x, y, z, yaw, size, confidence, v_x, v_y, acceleration_lon, yaw_deriv):
        # type: (int, int, float, float, float, float, ObjectSize, float, float, float, Union[float, None], Union[float, None]) -> None
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
        :param yaw_deriv: 0 for straight motion, positive for CCW (yaw increases), negative for CW
        """
        self.id = obj_id
        self._timestamp = timestamp
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw
        self.size = copy.copy(size)
        self.confidence = confidence
        self.v_x = v_x
        self.v_y = v_y
        self.road_localization = None
        self.rel_road_localization = None

        if acceleration_lon is not None:
            self.acceleration_lon = acceleration_lon
        else:
            raise NotImplementedError()

        if yaw_deriv is not None:
            self.yaw_deriv = yaw_deriv
        else:
            raise NotImplementedError()

    def predict(self, goal_timestamp, map_api):
        # type: (int, MapAPI) -> DynamicObject
        """
        Predict the object's location for the future timestamp
        !!! This function changes the object's location, velocity and timestamp !!!
        :param goal_timestamp: the goal timestamp for prediction
        :param lane_width: closest lane_width
        :return: predicted DynamicObject
        """
        pass


class EgoState(DynamicObject, DDSNonTypedMsg):
    def __init__(self, obj_id, timestamp, x, y, z, yaw, size, confidence,
                 v_x, v_y, acceleration_lon, yaw_deriv, steering_angle):
        # type: (int, int, float, float, float, float, ObjectSize, float, float, float, Union[float, None], Union[float, None], Union[float, None]) -> None
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
        :param yaw_deriv: radius of turning of the ego
        :param steering_angle: equivalent to knowing of turn_radius
        """
        DynamicObject.__init__(self, obj_id, timestamp, x, y, z, yaw, size, confidence, v_x, v_y,
                               acceleration_lon, yaw_deriv)
        if steering_angle is not None:
            self.steering_angle = steering_angle
        else:
            raise NotImplementedError()


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
        :return: predicted State
        """
        pass
