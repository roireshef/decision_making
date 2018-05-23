import copy
from typing import List
from typing import Optional

import numpy as np

from common_data.lcm.generatedFiles.gm_lcm import LcmDynamicObject
from common_data.lcm.generatedFiles.gm_lcm import LcmEgoState
from common_data.lcm.generatedFiles.gm_lcm import LcmNonTypedNumpyArray
from common_data.lcm.generatedFiles.gm_lcm import LcmObjectSize
from common_data.lcm.generatedFiles.gm_lcm import LcmOccupancyState
from common_data.lcm.generatedFiles.gm_lcm import LcmState
from common_data.lcm.generatedFiles.gm_lcm.LcmNumpyArray import LcmNumpyArray
from common_data.lcm.generatedFiles.gm_lcm import LcmMapState

from decision_making.src.exceptions import NoUniqueObjectStateForEvaluation
from decision_making.src.global_constants import PUBSUB_MSG_IMPL
from decision_making.src.planning.types import CartesianExtendedState, C_K, C_A
from decision_making.src.planning.types import CartesianState, C_X, C_Y, C_V, C_YAW
from decision_making.src.planning.types import FrenetState2D

from decision_making.src.planning.utils.lcm_utils import LCMUtils
from decision_making.src.planning.utils.map_utils import MapUtils
from mapping.src.service.map_service import MapService


class OccupancyState(PUBSUB_MSG_IMPL):
    ''' Members annotations for python 2 compliant classes '''
    timestamp = int
    free_space = np.ndarray
    confidence = np.ndarray

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

    def serialize(self):
        # type: () -> LcmOccupancyState
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
    def deserialize(cls, lcmMsg):
        # type: (LcmOccupancyState) -> OccupancyState
        return cls(lcmMsg.timestamp
                 , np.ndarray(shape = tuple(lcmMsg.free_space.shape)
                            , buffer = np.array(lcmMsg.free_space.data)
                            , dtype = float)
                 , np.ndarray(shape = tuple(lcmMsg.confidence.shape)
                            , buffer = np.array(lcmMsg.confidence.data)
                            , dtype = float))


class ObjectSize(PUBSUB_MSG_IMPL):
    ''' Members annotations for python 2 compliant classes '''
    length = float
    width = float
    height = float

    def __init__(self, length, width, height):
        # type: (float, float, float) -> None
        self.length = length
        self.width = width
        self.height = height

    def serialize(self):
        # type: () -> LcmObjectSize
        lcm_msg = LcmObjectSize()
        lcm_msg.length = self.length
        lcm_msg.width = self.width
        lcm_msg.height = self.height
        return lcm_msg

    @classmethod
    def deserialize(cls, lcmMsg):
        # type: (LcmObjectSize) -> ObjectSize
        return cls(lcmMsg.length, lcmMsg.width, lcmMsg.height)


# TODO: configure to initialize from segment_state and road_state ? (3 lazily-cached localizations)
class MapState(PUBSUB_MSG_IMPL):
    lane_state = FrenetState2D
    road_id = int
    segment_id = int
    lane_id = int

    def __init__(self, lane_state, road_id, segment_id, lane_id):
        # type: (FrenetState2D, int, int, int) -> MapState
        self.lane_state = lane_state
        self.road_id = road_id
        self.segment_id = segment_id
        self.lane_id = lane_id

    @property
    def road_state(self):
        return MapUtils.convert_lane_to_road_localization(self.lane_state, self.road_id, self.segment_id, self.lane_id)

    @property
    def segment_state(self):
        return MapUtils.convert_lane_to_segment_localization(self.lane_state, self.road_id, self.segment_id, self.lane_id)

    def serialize(self):
        # type: () -> LcmMapState
        lcm_msg = LcmMapState()
        lcm_msg.lane_state = LCMUtils.numpy_array_to_lcm_numpy_array(self.lane_state)
        lcm_msg.road_id = self.road_id
        lcm_msg.segment_id = self.segment_id
        lcm_msg.lane_id = self.lane_id
        return lcm_msg

    @classmethod
    def deserialize(cls, lcm_msg):
        # type: (LcmMapState) -> MapState
        return cls(np.ndarray(shape=tuple(lcm_msg.lane_state.shape)
                              , buffer=np.array(lcm_msg.lane_state.data)
                              , dtype=float)
        ,lcm_msg.road_id, lcm_msg.segment_id, lcm_msg.lane_id)


class NewDynamicObject(PUBSUB_MSG_IMPL):
    obj_id = int
    timestamp = int
    _cached_cartesian_state = CartesianExtendedState
    _cached_map_state = MapState
    size = ObjectSize
    confidence = float

    def __init__(self, obj_id, timestamp, cartesian_state, map_state, size, confidence):
        # type: (int, int, CartesianExtendedState, MapState, ObjectSize, float) -> NewDynamicObject
        """
        Data object that hold
        :param obj_id: object id
        :param timestamp: time of perception [nanosec.]
        :param cartesian_state: localization relative to map's cartesian origin frame
        :param map_state: localization in a map-object's frame (road,segment,lane)
        :param size: class ObjectSize
        :param confidence: of object's existence
        """
        self.obj_id = obj_id
        self.timestamp = timestamp
        self._cached_cartesian_state = cartesian_state
        self._cached_map_state = map_state
        self.size = copy.copy(size)
        self.confidence = confidence

    @property
    def x(self):
        return self.cartesian_state[C_X]

    @property
    def y(self):
        return self.cartesian_state[C_Y]

    @property
    def z(self):
        return 0

    @property
    def yaw(self):
        return self.cartesian_state[C_YAW]

    @property
    def velocity(self):
        return self.cartesian_state[C_V]

    @property
    def acceleration(self):
        return self.cartesian_state[C_A]

    @property
    def curvature(self):
        return self.cartesian_state[C_K]

    @property
    def cartesian_state(self):
        # type: () -> CartesianExtendedState
        if self._cached_cartesian_state is None:
            self._cached_cartesian_state = MapUtils.convert_map_to_cartesian_state(self._cached_map_state)
        return self._cached_cartesian_state

    @property
    def map_state(self):
        # type: () -> MapState
        if self._cached_map_state is None:
            self._cached_map_state = MapUtils.convert_cartesian_to_map_state(self._cached_cartesian_state)
        return self._cached_map_state

    @property
    def timestamp_in_sec(self):
        return self.timestamp * 1e-9

    @timestamp_in_sec.setter
    def timestamp_in_sec(self, value):
        self.timestamp = int(value * 1e9)

    @classmethod
    def create_from_cartesian_state(cls, obj_id: object, timestamp: object, cartesian_state: object, size: object, confidence: object):
        # type: (int, int, CartesianExtendedState, ObjectSize, float) -> NewDynamicObject
        """
        Constructor that gets only cartesian-state (without map-state)
        :param obj_id: object id
        :param timestamp: time of perception [nanosec.]
        :param cartesian_state: localization relative to map's cartesian origin frame
        :param size: class ObjectSize
        :param confidence: of object's existence
        """
        cls(obj_id, timestamp, cartesian_state, None, size, confidence)

    @classmethod
    def create_from_map_state(cls, obj_id, timestamp, map_state, size, confidence):
        # type: (int, int, MapState, ObjectSize, float) -> NewDynamicObject
        """
        Constructor that gets only map-state (without cartesian-state)
        :param obj_id: object id
        :param timestamp: time of perception [nanosec.]
        :param map_state: localization in a map-object's frame (road,segment,lane)
        :param size: class ObjectSize
        :param confidence: of object's existence
        """
        cls(obj_id, timestamp, None, map_state, size, confidence)

    def clone_from_cartesian_state(self, cartesian_state, timestamp=None):
        # type: (CartesianState, Optional[float]) -> NewDynamicObject
        """clones self while overriding cartesian_state and optionally timestamp"""
        return self.create_from_cartesian_state(self.obj_id, timestamp or self.timestamp, cartesian_state,
                                                self.size, self.confidence)

    def clone_from_map_state(self, map_state, timestamp=None):
        # type: (MapState, Optional[float]) -> NewDynamicObject
        """clones self while overriding map_state and optionally timestamp"""
        return self.create_from_map_state(self.obj_id, timestamp or self.timestamp, map_state,
                                          self.size, self.confidence)

    def serialize(self):
        # type: () -> LcmDynamicObject
        lcm_msg = LcmDynamicObject()
        lcm_msg.obj_id = self.obj_id
        lcm_msg.timestamp = self.timestamp
        lcm_msg._cached_cartesian_state = LCMUtils.numpy_array_to_lcm_numpy_array(self.cartesian_state)
        lcm_msg._cached_map_state = self.map_state.serialize()
        lcm_msg.size = self.size.serialize()
        lcm_msg.confidence = self.confidence
        return lcm_msg

    @classmethod
    def deserialize(cls, lcmMsg):
        # type: (LcmDynamicObject) -> NewDynamicObject
        return cls(lcmMsg.obj_id, lcmMsg.timestamp
                 , np.ndarray(shape=tuple(lcmMsg._cached_cartesian_state.shape)
                              , buffer=np.array(lcmMsg._cached_cartesian_state.data)
                              , dtype=float)
                    , MapState.deserialize(lcmMsg._cached_map_state)
                 , ObjectSize.deserialize(lcmMsg.size)
                 , lcmMsg.confidence)


class DynamicObject(PUBSUB_MSG_IMPL):
    ''' Members annotations for python 2 compliant classes '''
    obj_id = int
    timestamp = int
    x = float
    y = float
    z = float
    yaw = float
    size = ObjectSize
    confidence = float
    v_x = float
    v_y = float
    acceleration_lon = float
    omega_yaw = float

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
        self._cached_road_localization = None

    def clone_cartesian_state(self, timestamp_in_sec, cartesian_state):
        # type: (float, CartesianState) -> DynamicObject
        """
        Return a new DynamicObject instance with updated timestamp and cartesian state.
        Enables creating new instances of object from predicted trajectories.
        Assume that object's speed is only in the x axis
        :param timestamp_in_sec: global timestamp in [sec] of updated object
        :param cartesian_state: object cartesian state
        :return: Returns a new DynamicObject with updated state
        """

        timestamp = int(timestamp_in_sec * 1e9)
        x = cartesian_state[C_X]
        y = cartesian_state[C_Y]
        yaw = cartesian_state[C_YAW]

        # currently the velocity being used is self.total_speed (norm(v_x,v_y)) so v_y is important as well
        v_x = cartesian_state[C_V]

        # Fetch object's public fields
        object_fields = {k: v for k, v in self.__dict__.items() if k[0] != '_'}

        # Overwrite object fields
        object_fields['timestamp'] = timestamp
        object_fields['x'] = x
        object_fields['y'] = y
        object_fields['yaw'] = yaw
        object_fields['v_x'] = v_x

        # Construct a new object
        return self.__class__(**object_fields)

    @property
    # TODO: road localization needs to use frenet transformations
    def road_localization(self):
        # type: () -> MapState
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
    def road_longitudinal_speed(self):
        # type: () -> float
        """
        :return: Longitudinal speed (relative to road)
        """
        return self.total_speed * np.cos(self.road_localization.intra_road_yaw)

    @property
    #TODO: remove this when yaw issue is fixed
    def total_speed(self):
        # type: () -> float
        """
        Assuming no lateral slip
        :return: Total speed
        """
        return np.linalg.norm([self.v_x, self.v_y])

    @property
    def road_lateral_speed(self):
        # type: () -> float
        """
        :return: Lateral speed (relative to road)
        """
        return self.total_speed * np.sin(self.road_localization.intra_road_yaw)

    def serialize(self):
        # type: () -> LcmDynamicObject
        lcm_msg = LcmDynamicObject()
        lcm_msg.obj_id = self.obj_id
        lcm_msg.timestamp = self.timestamp
        lcm_msg.x = self.x
        lcm_msg.y = self.y
        lcm_msg.z = self.z
        lcm_msg.yaw = self.yaw
        lcm_msg.size = self.size.serialize()
        lcm_msg.confidence = self.confidence
        lcm_msg.v_x = self.v_x
        lcm_msg.v_y = self.v_y
        lcm_msg.acceleration_lon = self.acceleration_lon
        lcm_msg.omega_yaw = self.omega_yaw
        return lcm_msg

    @classmethod
    def deserialize(cls, lcmMsg):
        # type: (LcmDynamicObject) -> DynamicObject
        return cls(lcmMsg.obj_id, lcmMsg.timestamp
                 , lcmMsg.x, lcmMsg.y, lcmMsg.z, lcmMsg.yaw
                 , ObjectSize.deserialize(lcmMsg.size)
                 , lcmMsg.confidence, lcmMsg.v_x, lcmMsg.v_y
                 , lcmMsg.acceleration_lon, lcmMsg.omega_yaw)


class NewEgoState(NewDynamicObject):
    def __init__(self, obj_id, timestamp, cartesian_state, map_state, size, confidence):
        # type: (int, int, CartesianExtendedState, MapState, ObjectSize, float) -> NewEgoState

        """
        IMPORTANT! THE FIELDS IN THIS CLASS SHOULD NOT BE CHANGED ONCE THIS OBJECT IS INSTANTIATED

        Data object that hold
        :param obj_id: object id
        :param timestamp: time of perception [nanosec.]
        :param cartesian_state: localization relative to map's cartesian origin frame
        :param map_state: localization in a map-object's frame (road,segment,lane)
        :param size: class ObjectSize
        :param confidence: of object's existence
        """
        super(self.__class__, self).__init__(obj_id=obj_id, timestamp=timestamp, cartesian_state=cartesian_state, map_state=map_state, size=size, confidence=confidence)

    def serialize(self):
        # type: () -> LcmEgoState
        lcm_msg = LcmEgoState()
        lcm_msg.dynamic_obj = super(self.__class__, self).serialize()
        return lcm_msg

    @classmethod
    def deserialize(cls, lcmMsg):
        # type: (LcmEgoState) -> NewEgoState
        dyn_obj = NewDynamicObject.deserialize(lcmMsg.dynamic_obj)
        return cls(dyn_obj.obj_id, dyn_obj.timestamp
                 , dyn_obj._cached_cartesian_state, dyn_obj._cached_map_state
                 , dyn_obj.size
                 , dyn_obj.confidence)

#
# class EgoState(DynamicObject):
#     ''' Members annotations for python 2 compliant classes '''
#     obj_id = int
#     timestamp = int
#     x = float
#     y = float
#     z = float
#     yaw = float
#     size = ObjectSize
#     confidence = float
#     v_x = float
#     v_y = float
#     acceleration_lon = float
#     omega_yaw = float
#     steering_angle = float
#
#     def __init__(self, obj_id, timestamp, x, y, z, yaw, size, confidence,
#                  v_x, v_y, acceleration_lon, omega_yaw, steering_angle):
#         # type: (int, int, float, float, float, float, ObjectSize, float, float, float, float, float, float) -> None
#         """
#         IMPORTANT! THE FIELDS IN THIS CLASS SHOULD NOT BE CHANGED ONCE THIS OBJECT IS INSTANTIATED
#         :param obj_id:
#         :param timestamp:
#         :param x:
#         :param y:
#         :param z:
#         :param yaw:
#         :param size:
#         :param confidence:
#         :param v_x: velocity in ego's heading direction [m/sec]
#         :param v_y: velocity in ego's side (left) direction [m/sec]
#         :param acceleration_lon: in m/s^2
#         :param omega_yaw: radius of turning of the ego
#         :param steering_angle: equivalent to knowing of turn_radius
#         """
#         DynamicObject.__init__(self, obj_id, timestamp, x, y, z, yaw, size, confidence, v_x, v_y,
#                                acceleration_lon, omega_yaw)
#         self.steering_angle = steering_angle
#
#     @property
#     # TODO: change <length> to the distance between the two axles
#     # TODO: understand (w.r.t which axle counts) if we should use sin or tan here + validate vs sensor-alignments
#     def curvature(self):
#         """
#         For any point on a curve, the curvature measure is defined as 1/R where R is the radius length of a
#         circle that tangents the curve at that point. HERE, CURVATURE IS SIGNED (same sign as steering_angle).
#         For more information please see: https://en.wikipedia.org/wiki/Curvature#Curvature_of_plane_curves
#         """
#         return np.tan(self.steering_angle) / self.size.length
#
#     def serialize(self):
#         # type: () -> LcmEgoState
#         lcm_msg = LcmEgoState()
#         lcm_msg.dynamic_obj = super(self.__class__, self).serialize()
#         lcm_msg.steering_angle = self.steering_angle
#         return lcm_msg
#
#     @classmethod
#     def deserialize(cls, lcmMsg):
#         # type: (LcmEgoState) -> EgoState
#         dyn_obj = DynamicObject.deserialize(lcmMsg.dynamic_obj)
#         return cls(dyn_obj.obj_id, dyn_obj.timestamp
#                  , dyn_obj.x, dyn_obj.y, dyn_obj.z, dyn_obj.yaw
#                  , dyn_obj.size, dyn_obj.confidence
#                  , dyn_obj.v_x, dyn_obj.v_y, dyn_obj.acceleration_lon
#                  , dyn_obj.omega_yaw, lcmMsg.steering_angle)


class State(PUBSUB_MSG_IMPL):
    ''' Members annotations for python 2 compliant classes '''
    occupancy_state = OccupancyState
    dynamic_objects = List[NewDynamicObject]
    ego_state = NewEgoState

    def __init__(self, occupancy_state, dynamic_objects, ego_state):
        # type: (OccupancyState, List[NewDynamicObject], NewEgoState) -> None
        """
        main class for the world state. deep copy is required by self.clone_with!
        :param occupancy_state: free space
        :param dynamic_objects:
        :param ego_state:
        """
        self.occupancy_state = copy.deepcopy(occupancy_state)
        self.dynamic_objects = copy.deepcopy(dynamic_objects)
        self.ego_state = copy.deepcopy(ego_state)

    def clone_with(self, occupancy_state=None, dynamic_objects=None, ego_state=None):
        # type: (OccupancyState, List[NewDynamicObject], NewEgoState) -> State
        """
        clones state object with potential overriding of specific fields.
        requires deep-copying of all fields in State.__init__ !!
        """
        return State(occupancy_state or self.occupancy_state,
                     dynamic_objects or self.dynamic_objects,
                     ego_state or self.ego_state)

    def serialize(self):
        # type: () -> LcmState
        lcm_msg = LcmState()
        lcm_msg.occupancy_state = self.occupancy_state.serialize()
        ''' resize the list at once to the right length '''
        lcm_msg.num_obj = len(self.dynamic_objects)
        lcm_msg.dynamic_objects = list()
        for i in range(lcm_msg.num_obj):
            lcm_msg.dynamic_objects.append(self.dynamic_objects[i].serialize())
        lcm_msg.ego_state = self.ego_state.serialize()
        return lcm_msg

    @classmethod
    def deserialize(cls, lcmMsg):
        # type: (LcmState) -> State
        dynamic_objects = list()
        for i in range(lcmMsg.num_obj):
            dynamic_objects.append(NewDynamicObject.deserialize(lcmMsg.dynamic_objects[i]))
        ''' [DynamicObject.deserialize(lcmMsg.dynamic_objects[i]) for i in range(lcmMsg.num_obj)] '''
        return cls(OccupancyState.deserialize(lcmMsg.occupancy_state)
                 , dynamic_objects
                 , NewEgoState.deserialize(lcmMsg.ego_state))

    # TODO: remove when access to dynamic objects according to dictionary will be available.
    @classmethod
    def get_object_from_state(cls, state, target_obj_id):
        # type: (State, int) -> DynamicObject
        """
        Return the object with specific obj_id from world state
        :param state: the state to query
        :param target_obj_id: the id of the requested object
        :return: the dynamic_object matching the requested id
        """

        selected_objects = [obj for obj in state.dynamic_objects if obj.obj_id == target_obj_id]

        # Verify that object exists in state exactly once
        if len(selected_objects) != 1:
            raise NoUniqueObjectStateForEvaluation(
                'Found %d matching objects for object ID %d' % (len(selected_objects), target_obj_id))

        return selected_objects[0]

