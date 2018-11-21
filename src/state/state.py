import copy
from typing import List, Optional, Dict

import numpy as np

from common_data.interface.py.idl_generated_files.Rte_Types.sub_structures.LcmDynamicObject import LcmDynamicObject
from common_data.interface.py.idl_generated_files.Rte_Types.sub_structures.LcmEgoState import LcmEgoState
from common_data.interface.py.idl_generated_files.Rte_Types.sub_structures.LcmNonTypedSmallNumpyArray import LcmNonTypedSmallNumpyArray
from common_data.interface.py.idl_generated_files.Rte_Types.sub_structures.LcmObjectSize import LcmObjectSize
from common_data.interface.py.idl_generated_files.Rte_Types.sub_structures.LcmOccupancyState import LcmOccupancyState
from common_data.interface.py.idl_generated_files.Rte_Types.LcmState import LcmState
from common_data.interface.py.utils.serialization_utils import SerializationUtils

from decision_making.src.exceptions import MultipleObjectsWithRequestedID
from decision_making.src.global_constants import PUBSUB_MSG_IMPL, TIMESTAMP_RESOLUTION_IN_SEC
from decision_making.src.planning.behavioral.data_objects import RelativeLane
from decision_making.src.planning.types import C_X, C_Y, C_V, C_YAW, CartesianExtendedState, C_A, C_K, FrenetState2D
from decision_making.src.state.map_state import MapState
from decision_making.src.utils.map_utils import MapUtils


class OccupancyState(PUBSUB_MSG_IMPL):
    """ Members annotations for python 2 compliant classes """
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
        lcm_msg.free_space = SerializationUtils.serialize_non_typed_small_array(self.free_space)
        lcm_msg.confidence = SerializationUtils.serialize_non_typed_small_array(self.confidence)
        return lcm_msg

    @classmethod
    def deserialize(cls, lcmMsg):
        # type: (LcmOccupancyState) -> OccupancyState
        return cls(lcmMsg.timestamp,
                   SerializationUtils.deserialize_any_array(lcmMsg.free_space),
                   SerializationUtils.deserialize_any_array(lcmMsg.confidence))


class ObjectSize(PUBSUB_MSG_IMPL):
    """ Members annotations for python 2 compliant classes """
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


class DynamicObject(PUBSUB_MSG_IMPL):
    members_remapping = {'_cached_cartesian_state': 'cartesian_state',
                         '_cached_map_state': 'map_state'}

    obj_id = int
    timestamp = int
    _cached_cartesian_state = CartesianExtendedState
    _cached_map_states = Dict[RelativeLane, MapState]
    size = ObjectSize
    confidence = float

    def __init__(self, obj_id, timestamp, cartesian_state, map_state, size, confidence):
        # type: (int, int, CartesianExtendedState, MapState, ObjectSize, float) -> DynamicObject
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
        self._cached_map_states = {RelativeLane.SAME_LANE:map_state, RelativeLane.RIGHT_LANE:None, RelativeLane.LEFT_LANE:None}
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
            self._cached_cartesian_state = self._cached_map_states[RelativeLane.SAME_LANE].to_cartesian_state()
        return self._cached_cartesian_state

    @property
    def map_state(self):
        return self._get_adjacent_map_state(RelativeLane.SAME_LANE)

    @property
    def right_lane_map_state(self):
        """
        project the map_state on the right lane's Frenet frame (use caching)
        :return: projected map_state
        """
        return self._get_adjacent_map_state(RelativeLane.RIGHT_LANE)

    @property
    def left_lane_map_state(self):
        """
        project the map_state on the left lane's Frenet frame (use caching)
        :return: projected map_state
        """
        return self._get_adjacent_map_state(RelativeLane.LEFT_LANE)

    def _get_adjacent_map_state(self, relative_lane):
        # type: (RelativeLane) -> MapState
        """
        If relative_lane == RelativeLane.SAME_LANE, then get the same lane's map_state.
        Otherwise project the map_state on an adjacent lane's Frenet frame (use caching).
        If the adjacent lane does not exist, return empty MapState(None, None).
        :param relative_lane: either SAME_LANE or RIGHT_LANE or LEFT_LANE
        :return: projected map_state on the adjacent lane
        """
        if self._cached_map_states[relative_lane] is None:
            if relative_lane == RelativeLane.SAME_LANE:
                self._cached_map_states[relative_lane] = MapState.from_cartesian_state(self._cached_cartesian_state)
            else:
                adjacent_lane_ids = MapUtils.get_adjacent_lanes(self.map_state.lane_id, relative_lane)
                if len(adjacent_lane_ids) > 0:
                    frenet = MapUtils.get_lane_frenet_frame(adjacent_lane_ids[0])
                    # TODO: replace with faster implementation of cstate_to_fstate
                    fstate = frenet.cstate_to_fstate(self.cartesian_state)
                    self._cached_map_states[relative_lane] = MapState(fstate, adjacent_lane_ids[0])
                else:  # distinguish between not cached and non-existing lane
                    self._cached_map_states[relative_lane] = MapState(None, None)
        return self._cached_map_states[relative_lane]

    @staticmethod
    def sec_to_ticks(time_in_seconds: float):
        """
        Convert seconds to ticks (nanoseconds)
        :param time_in_seconds:
        :return: time in ticks (nanoseconds)
        """
        # type: float -> int
        return int(round(time_in_seconds / TIMESTAMP_RESOLUTION_IN_SEC))

    @staticmethod
    def ticks_to_sec(time_in_nanoseconds: int):
        """
        Convert ticks (nanoseconds) to seconds
        :param time_in_nanoseconds:
        :return: time in seconds
        """
        # type: int -> float
        return time_in_nanoseconds * TIMESTAMP_RESOLUTION_IN_SEC

    @property
    def timestamp_in_sec(self):
        return DynamicObject.ticks_to_sec(self.timestamp)

    @classmethod
    def create_from_cartesian_state(cls, obj_id, timestamp, cartesian_state, size, confidence):
        # type: (int, int, CartesianExtendedState, ObjectSize, float) -> DynamicObject
        """
        Constructor that gets only cartesian-state (without map-state)
        :param obj_id: object id
        :param timestamp: time of perception [nanosec.]
        :param cartesian_state: localization relative to map's cartesian origin frame
        :param size: class ObjectSize
        :param confidence: of object's existence
        """
        return cls(obj_id, timestamp, cartesian_state, None, size, confidence)

    @classmethod
    def create_from_map_state(cls, obj_id, timestamp, map_state, size, confidence):
        # type: (int, int, MapState, ObjectSize, float) -> DynamicObject
        """
        Constructor that gets only map-state (without cartesian-state)
        :param obj_id: object id
        :param timestamp: time of perception [nanosec.]
        :param map_state: localization in a map-object's frame (road,segment,lane)
        :param size: class ObjectSize
        :param confidence: of object's existence
        """
        return cls(obj_id, timestamp, None, map_state, size, confidence)

    def clone_from_cartesian_state(self, cartesian_state, timestamp_in_sec=None):
        # type: (CartesianExtendedState, Optional[float]) -> DynamicObject
        """clones self while overriding cartesian_state and optionally timestamp"""
        return self.__class__.create_from_cartesian_state(self.obj_id,
                                                          DynamicObject.sec_to_ticks(timestamp_in_sec or self.timestamp_in_sec),
                                                          cartesian_state,
                                                          self.size, self.confidence)

    def clone_from_map_state(self, map_state, timestamp_in_sec=None):
        # type: (MapState, Optional[float]) -> DynamicObject
        """clones self while overriding map_state and optionally timestamp"""
        return self.create_from_map_state(self.obj_id,
                                          DynamicObject.sec_to_ticks(timestamp_in_sec or self.timestamp_in_sec),
                                          map_state,
                                          self.size, self.confidence)

    def project_on_relative_lanes(self, relative_lanes):
        # type: (List[RelativeLane]) -> List[FrenetState2D]
        """
        Calculate frenet-states of the given object w.r.t. the relative (adjacent) lanes
        :param relative_lanes: list of relative lanes (same, left, right)
        :return: list of frenet-states of size len(relative_lanes)
        """
        return [self.map_state.lane_fstate if relative_lane == RelativeLane.SAME_LANE
                else self.left_lane_map_state.lane_fstate if relative_lane == RelativeLane.LEFT_LANE
                else self.right_lane_map_state.lane_fstate if relative_lane == RelativeLane.RIGHT_LANE else None
                for relative_lane in relative_lanes]

    def serialize(self):
        # type: () -> LcmDynamicObject
        lcm_msg = LcmDynamicObject()
        lcm_msg.obj_id = self.obj_id
        lcm_msg.timestamp = self.timestamp
        lcm_msg._cached_cartesian_state = self.cartesian_state
        lcm_msg._cached_map_state = self.map_state.serialize()
        lcm_msg.size = self.size.serialize()
        lcm_msg.confidence = self.confidence
        return lcm_msg

    @classmethod
    def deserialize(cls, lcmMsg):
        # type: (LcmDynamicObject) -> DynamicObject
        return cls(lcmMsg.obj_id, lcmMsg.timestamp
                   , lcmMsg._cached_cartesian_state.data
                   , MapState.deserialize(lcmMsg._cached_map_state)
                   , ObjectSize.deserialize(lcmMsg.size)
                   , lcmMsg.confidence)


class EgoState(DynamicObject):
    def __init__(self, obj_id, timestamp, cartesian_state, map_state, size, confidence):
        # type: (int, int, CartesianExtendedState, MapState, ObjectSize, float) -> EgoState
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
        super(self.__class__, self).__init__(obj_id=obj_id, timestamp=timestamp, cartesian_state=cartesian_state,
                                             map_state=map_state, size=size, confidence=confidence)

    def serialize(self):
        # type: () -> LcmEgoState
        lcm_msg = LcmEgoState()
        lcm_msg.dynamic_obj = super(self.__class__, self).serialize()
        return lcm_msg

    @classmethod
    def deserialize(cls, lcmMsg):
        # type: (LcmEgoState) -> EgoState
        dyn_obj = DynamicObject.deserialize(lcmMsg.dynamic_obj)
        return cls(dyn_obj.obj_id, dyn_obj.timestamp
                   , dyn_obj._cached_cartesian_state, dyn_obj._cached_map_states[RelativeLane.SAME_LANE]
                   , dyn_obj.size
                   , dyn_obj.confidence)


class State(PUBSUB_MSG_IMPL):
    """ Members annotations for python 2 compliant classes """
    occupancy_state = OccupancyState
    dynamic_objects = List[DynamicObject]
    ego_state = EgoState

    def __init__(self, occupancy_state, dynamic_objects, ego_state):
        # type: (OccupancyState, List[DynamicObject], EgoState) -> None
        """
        main class for the world state. deep copy is required by self.clone_with!
        :param occupancy_state: free space
        :param dynamic_objects:
        :param ego_state:
        """
        self.occupancy_state = occupancy_state
        self.dynamic_objects = dynamic_objects
        self.ego_state = ego_state

    def clone_with(self, occupancy_state=None, dynamic_objects=None, ego_state=None):
        # type: (OccupancyState, List[DynamicObject], EgoState) -> State
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

        for i in range(lcm_msg.num_obj):
            lcm_msg.dynamic_objects[i] = self.dynamic_objects[i].serialize()
        lcm_msg.ego_state = self.ego_state.serialize()
        return lcm_msg

    @classmethod
    def deserialize(cls, lcmMsg):
        # type: (LcmState) -> State
        dynamic_objects = list()
        for i in range(lcmMsg.num_obj):
            dynamic_objects.append(DynamicObject.deserialize(lcmMsg.dynamic_objects[i]))
        ''' [DynamicObject.deserialize(lcmMsg.dynamic_objects[i]) for i in range(lcmMsg.num_obj)] '''
        return cls(OccupancyState.deserialize(lcmMsg.occupancy_state)
                   , dynamic_objects
                   , EgoState.deserialize(lcmMsg.ego_state))

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
            raise MultipleObjectsWithRequestedID(
                'Found %d matching objects for object ID %d' % (len(selected_objects), target_obj_id))
        return selected_objects[0]

    # TODO: remove when access to dynamic objects according to dictionary will be available.
    @classmethod
    def get_objects_from_state(cls, state, target_obj_ids):
        # type: (State, List) -> List[DynamicObject]
        """
        Returns a list of object with the specific obj_ids from state
        :param state: the state to query
        :param target_obj_ids: a list of the id of the requested objects
        :return: the dynamic_objects matching the requested ids
        """
        selected_objects = [obj for obj in state.dynamic_objects if obj.obj_id in target_obj_ids]
        return selected_objects
