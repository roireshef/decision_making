import copy
from typing import List, Optional

import numpy as np

from decision_making.src.messages.dds_nontyped_message import DDSNonTypedMsg
from decision_making.src.messages.dds_typed_message import DDSTypedMsg
from mapping.src.model.localization import RoadLocalization

from decision_making.src.planning.types import CartesianState, C_X, C_Y, C_V, C_YAW
from mapping.src.service.map_service import MapService

from common_data.lcm.generatedFiles.gm_lcm import LcmNonTypedNumpyArray
from common_data.lcm.generatedFiles.gm_lcm import LcmOccupancyState
from common_data.lcm.generatedFiles.gm_lcm import LcmObjectSize
from common_data.lcm.generatedFiles.gm_lcm import LcmDynamicObject
from common_data.lcm.generatedFiles.gm_lcm import LcmEgoState
from common_data.lcm.generatedFiles.gm_lcm import LcmState


class OccupancyState(DDSTypedMsg):
    def __init__(self, timestamp: int, free_space: np.ndarray, confidence: np.ndarray):
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



class ObjectSize(DDSTypedMsg):
    def __init__(self, length: float, width: float, height: float):
        # type: (float, float, float) -> None
        self.length = length
        self.width = width
        self.height = height



class DynamicObject(DDSTypedMsg):
    def __init__(self, obj_id: int, timestamp: int, x: float, y: float, z: float, yaw: float, size: ObjectSize,
                 confidence: float, v_x: float, v_y: float, acceleration_lon: float, omega_yaw: float):
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

    def clone_cartesian_state(self, timestamp_in_sec: float, cartesian_state: CartesianState):
        """
        Return a new DynamicObject instance with updated timestamp and cartesian state.
        Enables creating new instances of object from predicted trajectories.
        Assume that object's speed is only in the x axis
        :param timestamp_in_sec: global timestamp in [sec] of updated object
        :param cartesian_state: object cartesian state
        :return: Returns a new DynamicObject with updated state
        """

        # TODO: set other attributes, as:
        # TODO: z coordinate, acceleration

        timestamp = int(timestamp_in_sec * 1e9)
        x = cartesian_state[C_X]
        y = cartesian_state[C_Y]
        z = 0.0
        yaw = cartesian_state[C_YAW]

        # Assume that object's speed is only in the x axis
        v_x = cartesian_state[C_V]
        v_y = 0.0

        # Fetch object's public fields
        object_fields = {k: v for k, v in self.__dict__.items() if k[0] != '_'}

        # Overwrite object fields
        object_fields['timestamp'] = timestamp
        object_fields['x'] = x
        object_fields['y'] = y
        object_fields['z'] = z
        object_fields['yaw'] = yaw
        object_fields['v_x'] = v_x
        object_fields['v_y'] = v_y

        # Construct a new object
        return self.__class__(**object_fields)

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


class EgoState(DynamicObject):
    def __init__(self, obj_id: int, timestamp: int, x: float, y: float, z: float, yaw: float, size: ObjectSize,
                 confidence: float,
                 v_x: float, v_y: float, acceleration_lon: float, omega_yaw: float, steering_angle: float):
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
    # TODO: change <length> to the distance between the two axles
    # TODO: understand (w.r.t which axle counts) if we should use sin or tan here + validate vs sensor-alignments
    def curvature(self):
        """
        For any point on a curve, the curvature measure is defined as 1/R where R is the radius length of a
        circle that tangents the curve at that point. HERE, CURVATURE IS SIGNED (same sign as steering_angle).
        For more information please see: https://en.wikipedia.org/wiki/Curvature#Curvature_of_plane_curves
        """
        return np.tan(self.steering_angle) / self.size.length

class State(DDSTypedMsg):
    def __init__(self, occupancy_state: OccupancyState, dynamic_objects: List[DynamicObject], ego_state: EgoState):
        # type: (OccupancyState, List[DynamicObject], EgoState) -> None
        """
        main class for the world state. deep copy is required by self.clone_with!
        :param occupancy_state: free space
        :param dynamic_objects:
        :param ego_state:
        """
        self.occupancy_state = copy.deepcopy(occupancy_state)
        self.dynamic_objects = copy.deepcopy(dynamic_objects)
        self.ego_state = copy.deepcopy(ego_state)

    def clone_with(self, occupancy_state: Optional[OccupancyState] = None,
                   dynamic_objects: Optional[List[DynamicObject]] = None,
                   ego_state: Optional[EgoState] = None):
        """
        clones state object with potential overriding of specific fields.
        requires deep-copying of all fields in State.__init__ !!
        """
        return State(occupancy_state or self.occupancy_state,
                     dynamic_objects or self.dynamic_objects,
                     ego_state or self.ego_state)
