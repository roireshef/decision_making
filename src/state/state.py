import copy
import numpy as np
from common_data.interface.Rte_Types.python.sub_structures.TsSYS_DynamicObject import TsSYSDynamicObject
from common_data.interface.Rte_Types.python.sub_structures.TsSYS_EgoState import TsSYSEgoState
from common_data.interface.Rte_Types.python.sub_structures.TsSYS_ObjectSize import TsSYSObjectSize
from common_data.interface.Rte_Types.python.sub_structures.TsSYS_OccupancyState import TsSYSOccupancyState
from common_data.interface.Rte_Types.python.sub_structures.TsSYS_State import TsSYSState
from common_data.interface.py.utils.serialization_utils import SerializationUtils
from decision_making.src.exceptions import MultipleObjectsWithRequestedID
from decision_making.src.global_constants import PUBSUB_MSG_IMPL, TIMESTAMP_RESOLUTION_IN_SEC
from decision_making.src.planning.types import C_X, C_Y, C_V, C_YAW, CartesianExtendedState, C_A, C_K
from decision_making.src.state.map_state import MapState
from decision_making.src.utils.map_utils import MapUtils
from typing import List, Optional


class OccupancyState(PUBSUB_MSG_IMPL):
    e_Cnt_Timestamp = int
    s_FreeSpace = np.ndarray
    s_Confidence = np.ndarray

    def __init__(self, timestamp: int, free_space: np.ndarray, confidence: np.ndarray):
        """
        free space description
        :param timestamp of free space
        :param free_space: array of directed segments defines a free space border
        :param confidence: array per segment
        """
        self.e_Cnt_Timestamp = timestamp
        self.s_FreeSpace = np.copy(free_space)
        self.s_Confidence = np.copy(confidence)

    def serialize(self) ->TsSYSOccupancyState:
        pubsub_msg = TsSYSOccupancyState()
        pubsub_msg.e_Cnt_Timestamp = self.e_Cnt_Timestamp
        pubsub_msg.s_FreeSpace = SerializationUtils.serialize_non_typed_small_array(self.s_FreeSpace)
        pubsub_msg.s_Confidence = SerializationUtils.serialize_non_typed_small_array(self.s_Confidence)
        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSOccupancyState) -> ():
        return cls(pubsubMsg.e_Cnt_Timestamp,
                   SerializationUtils.deserialize_any_array(pubsubMsg.s_FreeSpace),
                   SerializationUtils.deserialize_any_array(pubsubMsg.s_Confidence))


class ObjectSize(PUBSUB_MSG_IMPL):
    e_l_Length = float
    e_l_Width = float
    e_l_Height = float

    def __init__(self, length: float, width: float, height: float) -> None:
        self.e_l_Length = length
        self.e_l_Width = width
        self.e_l_Height = height

    def serialize(self):
        # type: () -> TsSYSObjectSize
        pubsub_msg = TsSYSObjectSize()
        pubsub_msg.e_l_Length = self.e_l_Length
        pubsub_msg.e_l_Width = self.e_l_Width
        pubsub_msg.e_l_Height = self.e_l_Height
        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg):
        # type: (TsSYSObjectSize) -> ObjectSize
        return cls(pubsubMsg.e_l_Length, pubsubMsg.e_l_Width, pubsubMsg.e_l_Height)


class DynamicObject(PUBSUB_MSG_IMPL):
    members_remapping = {'_cached_cartesian_state': 'cartesian_state',
                         '_cached_map_state': 'map_state'}

    e_i_ObjectID = int
    e_Cnt_Timestamp = int
    _cached_cartesian_state = CartesianExtendedState
    _cached_map_state = MapState
    s_Size = ObjectSize
    e_r_Confidence = float

    def __init__(self, obj_id, timestamp, cartesian_state, map_state, size, confidence):
        # type: (int, int, Optional[CartesianExtendedState], Optional[MapState], ObjectSize, float) -> None
        """
        Data object that hold
        :param obj_id: object id
        :param timestamp: time of perception [nanosec.]
        :param cartesian_state: localization relative to map's cartesian origin frame
        :param map_state: localization in a map-object's frame (road,segment,lane)
        :param size: class ObjectSize
        :param confidence: of object's existence
        """
        self.e_i_ObjectID = obj_id
        self.e_Cnt_Timestamp = timestamp
        self._cached_cartesian_state = cartesian_state
        self._cached_map_state = map_state
        self.s_Size = copy.copy(size)
        self.e_r_Confidence = confidence

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
            lane_frenet = MapUtils.get_lane_frenet_frame(self.map_state.e_i_LaneID)
            self._cached_cartesian_state = lane_frenet.fstate_to_cstate(self.map_state.a_LaneFState)
        return self._cached_cartesian_state

    @property
    def map_state(self):
        # type: () -> MapState
        if self._cached_map_state is None:
            closest_lane_id = MapUtils.get_closest_lane(self.cartesian_state[:(C_Y+1)])
            lane_frenet = MapUtils.get_lane_frenet_frame(closest_lane_id)
            self._cached_map_state = MapState(lane_frenet.cstate_to_fstate(self.cartesian_state), closest_lane_id)
        return self._cached_map_state

    @staticmethod
    def sec_to_ticks(time_in_seconds):
        # type: (float) -> int
        """
        Convert seconds to ticks (nanoseconds)
        :param time_in_seconds:
        :return: time in ticks (nanoseconds)
        """
        return int(round(time_in_seconds / TIMESTAMP_RESOLUTION_IN_SEC))

    @staticmethod
    def ticks_to_sec(time_in_nanoseconds):
        # type: (int) -> float
        """
        Convert ticks (nanoseconds) to seconds
        :param time_in_nanoseconds:
        :return: time in seconds
        """
        return time_in_nanoseconds * TIMESTAMP_RESOLUTION_IN_SEC

    @property
    def timestamp_in_sec(self):
        return DynamicObject.ticks_to_sec(self.e_Cnt_Timestamp)

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
        return self.__class__.create_from_cartesian_state(self.e_i_ObjectID,
                                                          DynamicObject.sec_to_ticks(timestamp_in_sec or self.timestamp_in_sec),
                                                          cartesian_state,
                                                          self.s_Size, self.e_r_Confidence)

    def clone_from_map_state(self, map_state, timestamp_in_sec=None):
        # type: (MapState, Optional[float]) -> DynamicObject
        """clones self while overriding map_state and optionally timestamp"""
        return self.create_from_map_state(self.e_i_ObjectID,
                                          DynamicObject.sec_to_ticks(timestamp_in_sec or self.timestamp_in_sec),
                                          map_state,
                                          self.s_Size, self.e_r_Confidence)

    def serialize(self):
        # type: () -> TsSYSDynamicObject
        pubsub_msg = TsSYSDynamicObject()
        pubsub_msg.e_i_ObjectID = self.e_i_ObjectID
        pubsub_msg.e_Cnt_Timestamp = self.e_Cnt_Timestamp
        pubsub_msg._cached_cartesian_state = self.cartesian_state
        pubsub_msg._cached_map_state = self._cached_map_state.serialize()
        pubsub_msg._cached_map_state_on_host_lane = self._cached_map_state.serialize()
        pubsub_msg.s_Size = self.s_Size.serialize()
        pubsub_msg.e_r_Confidence = self.e_r_Confidence
        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg):
        # type: (TsSYSDynamicObject) -> DynamicObject
        return cls(pubsubMsg.e_i_ObjectID, pubsubMsg.e_Cnt_Timestamp
                   , pubsubMsg._cached_cartesian_state
                   , MapState.deserialize(pubsubMsg._cached_map_state) if pubsubMsg._cached_map_state.lane_id > 0 else None
                   , ObjectSize.deserialize(pubsubMsg.s_Size)
                   , pubsubMsg.e_r_Confidence)


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
        # type: () -> TsSYSEgoState
        pubsub_msg = TsSYSEgoState()
        pubsub_msg.dynamic_obj = super(self.__class__, self).serialize()
        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg):
        # type: (TsSYSEgoState) -> EgoState
        dyn_obj = DynamicObject.deserialize(pubsubMsg.s_DynamicObject)
        return cls(dyn_obj.e_i_ObjectID, dyn_obj.e_Cnt_Timestamp
                   , dyn_obj._cached_cartesian_state
                   , dyn_obj._cached_map_state
                   , dyn_obj.s_Size
                   , dyn_obj.e_r_Confidence)


class State(PUBSUB_MSG_IMPL):
    """ Members annotations for python 2 compliant classes """
    s_OccupancyState = OccupancyState
    s_DynamicObjects = List[DynamicObject]
    s_EgoState = EgoState

    def __init__(self, occupancy_state: OccupancyState, dynamic_objects: List[DynamicObject], ego_state: EgoState)-> None:
        """
        main class for the world state. deep copy is required by self.clone_with!
        :param occupancy_state: free space
        :param dynamic_objects:
        :param ego_state:
        """
        self.s_OccupancyState = occupancy_state
        self.s_DynamicObjects = dynamic_objects
        self.s_EgoState = ego_state

    def clone_with(self, occupancy_state: OccupancyState = None, dynamic_objects: List[DynamicObject] = None,
                   ego_state: EgoState = None):
        """
        clones state object with potential overriding of specific fields.
        requires deep-copying of all fields in State.__init__ !!
        """
        return State(occupancy_state or self.s_OccupancyState,
                     dynamic_objects if dynamic_objects is not None else self.s_DynamicObjects,
                     ego_state or self.s_EgoState)

    def serialize(self):
        # type: () -> TsSYSState
        pubsub_msg = TsSYSState()
        pubsub_msg.s_OccupancyState = self.s_OccupancyState.serialize()
        ''' resize the list at once to the right length '''
        pubsub_msg.e_Cnt_NumOfObjects = len(self.s_DynamicObjects)

        for i in range(pubsub_msg.e_Cnt_NumOfObjects):
            pubsub_msg.s_DynamicObjects[i] = self.s_DynamicObjects[i].serialize()
        pubsub_msg.s_EgoState = self.s_EgoState.serialize()
        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg):
        # type: (TsSYSState) -> State
        dynamic_objects = list()
        for i in range(pubsubMsg.e_Cnt_NumOfObjects):
            dynamic_objects.append(DynamicObject.deserialize(pubsubMsg.s_DynamicObjects[i]))
        return cls(OccupancyState.deserialize(pubsubMsg.s_OccupancyState)
                   , dynamic_objects
                   , EgoState.deserialize(pubsubMsg.s_EgoState))

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
