import copy
from typing import List, Optional

import numpy as np

from mapping.src.model.localization import RoadLocalization
from mapping.src.service.map_service import MapService

from common_data.lcm.generatedFiles.gm_lcm import LcmNonTypedNumpyArray
from common_data.lcm.generatedFiles.gm_lcm import LcmOccupancyState
from common_data.lcm.generatedFiles.gm_lcm import LcmObjectSize
from common_data.lcm.generatedFiles.gm_lcm import LcmDynamicObject
from common_data.lcm.generatedFiles.gm_lcm import LcmEgoState
from common_data.lcm.generatedFiles.gm_lcm import LcmState


class OccupancyState:
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

    def to_lcm(self) -> LcmOccupancyState:
        lcm_msg = LcmOccupancyState()
        lcm_msg.timestamp = self.timestamp
        lcm_msg.free_space = LcmNonTypedNumpyArray()
        lcm_msg.free_space.num_dimensions = len(self.free_space.shape)
        lcm_msg.free_space.shape = list(self.free_space.shape)
        lcm_msg.free_space.length = self.free_space.size
        lcm_msg.free_space.data = self.free_space.flat.__array__().tolist()
        lcm_msg.confidence = LcmNonTypedNumpyArray()
        lcm_msg.confidence.num_dimensions = len(self.confidence.shape)
        lcm_msg.confidence.shape = list(self.confidence.shape)
        lcm_msg.confidence.length = self.confidence.size
        lcm_msg.confidence.data = self.confidence.flat.__array__().tolist()
        return lcm_msg

    @classmethod
    def from_lcm(cls, lcmMsg: LcmOccupancyState):
        return cls(lcmMsg.timestamp
                 , np.ndarray(shape = tuple(lcmMsg.free_space.shape)
                            , buffer = np.array(lcmMsg.free_space.data)
                            , dtype = float)
                 , np.ndarray(shape = tuple(lcmMsg.confidence.shape)
                            , buffer = np.array(lcmMsg.confidence.data)
                            , dtype = float))


class ObjectSize:
    def __init__(self, length, width, height):
        # type: (float, float, float) -> None
        self.length = length
        self.width = width
        self.height = height

    def to_lcm(self) -> LcmObjectSize:
        lcm_msg = LcmObjectSize()
        lcm_msg.length = self.length
        lcm_msg.width = self.width
        lcm_msg.height = self.height
        return lcm_msg

    @classmethod
    def from_lcm(cls, lcmMsg: LcmObjectSize):
        return cls(lcmMsg.length, lcmMsg.width, lcmMsg.height)


class DynamicObject:
    def __init__(self, obj_id, timestamp, x, y, z, yaw, size, confidence, v_x, v_y, acceleration_lon, omega_yaw):
        # type: (int, int, float, float, float, float, ObjectSize, float, float, float, float, float) -> DynamicObject
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

    def to_lcm(self) -> LcmDynamicObject:
        lcm_msg = LcmDynamicObject()
        lcm_msg.obj_id = self.obj_id
        lcm_msg.timestamp = self.timestamp
        lcm_msg.x = self.x
        lcm_msg.y = self.y
        lcm_msg.z = self.z
        lcm_msg.yaw = self.yaw
        lcm_msg.size = self.size.to_lcm()
        lcm_msg.confidence = self.confidence
        lcm_msg.v_x = self.v_x
        lcm_msg.v_y = self.v_y
        lcm_msg.acceleration_lon = self.acceleration_lon
        lcm_msg.omega_yaw = self.omega_yaw
        return lcm_msg

    @classmethod
    def from_lcm(cls, lcmMsg: LcmDynamicObject):
        return cls(lcmMsg.obj_id, lcmMsg.timestamp
                 , lcmMsg.x, lcmMsg.y, lcmMsg.z, lcmMsg.yaw
                 , ObjectSize.from_lcm(lcmMsg.size)
                 , lcmMsg.confidence, lcmMsg.v_x, lcmMsg.v_y
                 , lcmMsg.acceleration_lon, lcmMsg.omega_yaw)


class EgoState(DynamicObject):
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

    def to_lcm(self) -> LcmEgoState:
        lcm_msg = LcmEgoState()
        lcm_msg.dynamic_obj = super(self.__class__, self).to_lcm()
        lcm_msg.steering_angle = self.steering_angle
        return lcm_msg

    @classmethod
    def from_lcm(cls, lcmMsg: LcmEgoState):
        dyn_obj = DynamicObject.from_lcm(lcmMsg.dynamic_obj)
        return cls(dyn_obj.obj_id, dyn_obj.timestamp
                 , dyn_obj.x, dyn_obj.y, dyn_obj.z, dyn_obj.yaw
                 , dyn_obj.size, dyn_obj.confidence
                 , dyn_obj.v_x, dyn_obj.v_y, dyn_obj.acceleration_lon
                 , dyn_obj.omega_yaw, lcmMsg.steering_angle)


class State:
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

    def to_lcm(self) -> LcmState:
        lcm_msg = LcmState()
        lcm_msg.occupancy_state = self.occupancy_state.to_lcm()
        ''' resize the list at once to the right length '''
        lcm_msg.num_obj = len(self.dynamic_objects)
        #lcm_msg.dynamic_objects = [LcmDynamicObject() for i in range(lcm_msg.num_obj)]
        lcm_msg.dynamic_objects = list()
        for i in range(lcm_msg.num_obj):
            #lcm_msg.dynamic_objects[i] = self.dynamic_objects[i].to_lcm()
            lcm_msg.dynamic_objects.append(self.dynamic_objects[i].to_lcm())
        lcm_msg.ego_state = self.ego_state.to_lcm()
        return lcm_msg

    @classmethod
    def from_lcm(cls, lcmMsg: LcmState):
        dynamic_objects = list()
        for i in range(lcmMsg.num_obj):
            dynamic_objects.append(DynamicObject.from_lcm(lcmMsg.dynamic_objects[i]))
        ''' [DynamicObject.from_lcm(lcmMsg.dynamic_objects[i]) for i in range(lcmMsg.num_obj)] '''
        return cls(OccupancyState.from_lcm(lcmMsg.occupancy_state)
                 , dynamic_objects
                 , EgoState.from_lcm(lcmMsg.ego_state))

