import copy
from typing import List, Optional

import numpy as np

from decision_making.src.messages.dds_nontyped_message import DDSNonTypedMsg
from mapping.src.model.localization import RoadLocalization
from mapping.src.service.map_service import MapService


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
    def __init__(self, obj_id, timestamp, x, y, z, yaw, size, confidence, v_x, v_y, acceleration_lon, omega_yaw):
        # type: (int, int, float, float, float, float, ObjectSize, float, float, float, float, float) -> None
        """
        IMPORTANT! THE FIELDS IN THIS CLASS SHOULD NOT BE CHANGED ONCE THIS OBJECT IS INSTANTIATED
        both ego and other dynamic objects
        :param obj_id: object id
        :param timestamp: time of perception
        :param x: for ego in world coordinates, for the rest relatively to ego
        :param y:
        :param z:
        :param yaw: for ego 0 means along X axis, for the rest 0 means forward direction relatively to ego
        :param size: class ObjectSize
        :param confidence: of object's existence
        :param v_x: velocity in object's heading direction [m/sec]
        :param v_y: velocity in object's side (left) direction [m/sec] (usually close to zero)
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
        self.acceleration_lon = acceleration_lon
        self.omega_yaw = omega_yaw
        self._cached_road_localization: Optional[RoadLocalization] = None

    @property
    def road_localization(self):
        # type: () -> RoadLocalization
        if self._cached_road_localization is None:
            self._cached_road_localization = MapService.get_instance().compute_road_localization(
                np.array([self.x, self.y, self.z]), self.yaw)
        return self._cached_road_localization

    @property
    def timestamp_in_sec(self):
        return self.timestamp * 1e-9

    @timestamp_in_sec.setter
    def timestamp_in_sec(self, value):
        self.timestamp = int(value * 1e9)

    @property
    def road_longitudinal_speed(self) -> float:
        """
        Assuming no lateral slip
        :return: Longitudinal speed (relative to road)
        """
        return np.linalg.norm([self.v_x, self.v_y]) * np.cos(self.road_localization.intra_road_yaw)

    @property
    def road_lateral_speed(self) -> float:
        """
        Assuming no lateral slip
        :return: Longitudinal speed (relative to road)
        """
        return np.linalg.norm([self.v_x, self.v_y]) * np.sin(self.road_localization.intra_road_yaw)


class EgoState(DynamicObject, DDSNonTypedMsg):
    def __init__(self, obj_id, timestamp, x, y, z, yaw, size, confidence,
                 v_x, v_y, acceleration_lon, omega_yaw, steering_angle):
        # type: (int, int, float, float, float, float, ObjectSize, float, float, float, float, float, float) -> None
        """
        IMPORTANT! THE FIELDS IN THIS CLASS SHOULD NOT BE CHANGED ONCE THIS OBJECT IS INSTANTIATED
        :param obj_id:
        :param timestamp:
        :param x:
        :param y:
        :param z:
        :param yaw:
        :param size:
        :param confidence:
        :param v_x: velocity in ego's heading direction [m/sec]
        :param v_y: velocity in ego's side (left) direction [m/sec]
        :param acceleration_lon: in m/s^2
        :param omega_yaw: radius of turning of the ego
        :param steering_angle: equivalent to knowing of turn_radius
        """
        DynamicObject.__init__(self, obj_id, timestamp, x, y, z, yaw, size, confidence, v_x, v_y,
                               acceleration_lon, omega_yaw)
        self.steering_angle = steering_angle

    @property
    def curvature(self):    # curvature is signed (same sign as steering_angle)
        # TODO: change <length> to the distance between the two axles
        return np.tan(self.steering_angle) / self.size.length


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
