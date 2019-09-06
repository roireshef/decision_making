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
    timestamp = int
    free_space = np.ndarray
    confidence = np.ndarray

    def __init__(self, timestamp: int, free_space: np.ndarray, confidence: np.ndarray):
        """
        free space description
        :param timestamp of free space
        :param free_space: array of directed segments defines a free space border
        :param confidence: array per segment
        """
        self.timestamp = timestamp
        self.free_space = np.copy(free_space)
        self.confidence = np.copy(confidence)

    def serialize(self) ->TsSYSOccupancyState:
        pubsub_msg = TsSYSOccupancyState()
        pubsub_msg.e_Cnt_Timestamp = self.timestamp
        pubsub_msg.s_FreeSpace = SerializationUtils.serialize_non_typed_small_array(self.free_space)
        pubsub_msg.s_Confidence = SerializationUtils.serialize_non_typed_small_array(self.confidence)
        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSOccupancyState) -> ():
        return cls(pubsubMsg.e_Cnt_Timestamp,
                   SerializationUtils.deserialize_any_array(pubsubMsg.s_FreeSpace),
                   SerializationUtils.deserialize_any_array(pubsubMsg.s_Confidence))


class ObjectSize(PUBSUB_MSG_IMPL):
    length = float
    width = float
    height = float

    def __init__(self, length: float, width: float, height: float) -> None:
        self.length = length
        self.width = width
        self.height = height

    def serialize(self):
        # type: () -> TsSYSObjectSize
        pubsub_msg = TsSYSObjectSize()
        pubsub_msg.e_l_Length = self.length
        pubsub_msg.e_l_Width = self.width
        pubsub_msg.e_l_Height = self.height
        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg):
        # type: (TsSYSObjectSize) -> ObjectSize
        return cls(pubsubMsg.e_l_Length, pubsubMsg.e_l_Width, pubsubMsg.e_l_Height)


class DynamicObject(PUBSUB_MSG_IMPL):
    members_remapping = {'_cached_cartesian_state': 'cartesian_state',
                         '_cached_map_state': 'map_state'}

    obj_id = int
    timestamp = int
    _cached_cartesian_state = CartesianExtendedState
    _cached_map_state = MapState
    size = ObjectSize
    confidence = float
    off_map = bool

    def __init__(self, obj_id, timestamp, cartesian_state, map_state, size, confidence, off_map):
        # type: (int, int, Optional[CartesianExtendedState], Optional[MapState], ObjectSize, float, bool) -> None
        """
        Data object that hold
        :param obj_id: object id
        :param timestamp: time of perception [nanosec.]
        :param cartesian_state: localization relative to map's cartesian origin frame
        :param map_state: localization in a map-object's frame (road,segment,lane)
        :param size: class ObjectSize
        :param confidence: of object's existence
        :param off_map: indicates if the vehicle is off the map
        """
        self.obj_id = obj_id
        self.timestamp = timestamp
        self._cached_cartesian_state = cartesian_state
        self._cached_map_state = map_state
        self.size = copy.copy(size)
        self.confidence = confidence
        self.off_map = off_map

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
            lane_frenet = MapUtils.get_lane_frenet_frame(self.map_state.lane_id)
            self._cached_cartesian_state = lane_frenet.fstate_to_cstate(self.map_state.lane_fstate)
        return self._cached_cartesian_state

    def bounding_box(self):
        """
        Gets the cartesian coordinates of the four corners of the object's bounding box
        :return: [front left, front right, rear right, rear left]
        [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        """
        # create matrix of homogeneous points, assuming zero heading. Each coordinate is a column vector
        bbox = np.array([[self.x - self.size.width/2.0, self.y + self.size.length/2.0, 1],
                         [self.x + self.size.width/2.0, self.y + self.size.length/2.0, 1],
                         [self.x + self.size.width/2.0, self.y - self.size.length/2.0, 1],
                         [self.x - self.size.width/2.0, self.y - self.size.length/2.0, 1]]).transpose()
        # shift so that the center will be at [0,0]
        translate_mat = np.array([[1, 0, -self.x],
                                  [0, 1, -self.y],
                                  [0, 0,  1]])

        # rotate about the origin using the heading
        # use clockwise rotation matrix since positive heading is clockwise
        heading = self.yaw
        rotation_mat = np.array([[np.cos(heading), np.sin(heading), 0],
                                 [-np.sin(heading), np.cos(heading),  0],
                                 [0,               0,                1]])

        # translate back to original point
        untranslate_mat = np.array([[1, 0, self.x],
                                    [0, 1, self.y],
                                    [0, 0, 1]])

        # apply transformations to points
        bbox = untranslate_mat.dot(rotation_mat).dot(translate_mat).dot(bbox)

        # transpose matrix to get an array of coordinates
        bbox = bbox.transpose()

        # normalize and remove homogenous 3rd coordinate
        for i in range(len(bbox)):
            bbox[i] = bbox[i] / bbox[i,2]
        return bbox[:,0:2]


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
        return DynamicObject.ticks_to_sec(self.timestamp)

    @classmethod
    def create_from_cartesian_state(cls, obj_id, timestamp, cartesian_state, size, confidence, off_map):
        # type: (int, int, CartesianExtendedState, ObjectSize, float, bool) -> DynamicObject
        """
        Constructor that gets only cartesian-state (without map-state)
        :param obj_id: object id
        :param timestamp: time of perception [nanosec.]
        :param cartesian_state: localization relative to map's cartesian origin frame
        :param size: class ObjectSize
        :param confidence: of object's existence
        :param off_map: indicates if the vehicle is off the map

        """
        return cls(obj_id, timestamp, cartesian_state, None, size, confidence, off_map)

    @classmethod
    def create_from_map_state(cls, obj_id, timestamp, map_state, size, confidence, off_map):
        # type: (int, int, MapState, ObjectSize, float, bool) -> DynamicObject
        """
        Constructor that gets only map-state (without cartesian-state)
        :param obj_id: object id
        :param timestamp: time of perception [nanosec.]
        :param map_state: localization in a map-object's frame (road,segment,lane)
        :param size: class ObjectSize
        :param confidence: of object's existence
        :param off_map: is the vehicle is off the map
        """
        return cls(obj_id, timestamp, None, map_state, size, confidence, off_map)

    def clone_from_cartesian_state(self, cartesian_state, timestamp_in_sec=None):
        # type: (CartesianExtendedState, Optional[float]) -> DynamicObject
        """clones self while overriding cartesian_state and optionally timestamp"""
        return self.__class__.create_from_cartesian_state(self.obj_id,
                                                          DynamicObject.sec_to_ticks(timestamp_in_sec or self.timestamp_in_sec),
                                                          cartesian_state,
                                                          self.size, self.confidence, self.off_map)

    def clone_from_map_state(self, map_state, timestamp_in_sec=None):
        # type: (MapState, Optional[float]) -> DynamicObject
        """clones self while overriding map_state and optionally timestamp"""
        return self.create_from_map_state(self.obj_id,
                                          DynamicObject.sec_to_ticks(timestamp_in_sec or self.timestamp_in_sec),
                                          map_state,
                                          self.size, self.confidence, self.off_map)

    def serialize(self):
        # type: () -> TsSYSDynamicObject
        pubsub_msg = TsSYSDynamicObject()
        pubsub_msg.e_i_ObjectID = self.obj_id
        pubsub_msg.e_Cnt_Timestamp = self.timestamp
        pubsub_msg.a_e_CachedCartesianState = self.cartesian_state
        pubsub_msg.s_CachedMapState = self._cached_map_state.serialize()
        pubsub_msg.s_Size = self.size.serialize()
        pubsub_msg.e_r_Confidence = self.confidence
        pubsub_msg.e_b_offMap = self.off_map
        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg):
        # type: (TsSYSDynamicObject) -> DynamicObject
        return cls(pubsubMsg.e_i_ObjectID, pubsubMsg.e_Cnt_Timestamp
                   , pubsubMsg.a_e_CachedCartesianState
                   , MapState.deserialize(pubsubMsg.s_CachedMapState) if pubsubMsg.s_CachedMapState.e_i_LaneID > 0 else None
                   , ObjectSize.deserialize(pubsubMsg.s_Size)
                   , pubsubMsg.e_r_Confidence, pubsubMsg.e_b_offMap)


class EgoState(DynamicObject):
    def __init__(self, obj_id, timestamp, cartesian_state, map_state, size, confidence, off_map):
        # type: (int, int, CartesianExtendedState, MapState, ObjectSize, float, bool) -> EgoState
        """
        IMPORTANT! THE FIELDS IN THIS CLASS SHOULD NOT BE CHANGED ONCE THIS OBJECT IS INSTANTIATED

        Data object that hold
        :param obj_id: object id
        :param timestamp: time of perception [nanosec.]
        :param cartesian_state: localization relative to map's cartesian origin frame
        :param map_state: localization in a map-object's frame (road,segment,lane)
        :param size: class ObjectSize
        :param confidence: of object's existence
        :param off_map: indicates if the object is off map
        """
        super(self.__class__, self).__init__(obj_id=obj_id, timestamp=timestamp, cartesian_state=cartesian_state,
                                             map_state=map_state, size=size, confidence=confidence, off_map=off_map)

    def serialize(self):
        # type: () -> TsSYSEgoState
        pubsub_msg = TsSYSEgoState()
        pubsub_msg.s_DynamicObject = super(self.__class__, self).serialize()
        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg):
        # type: (TsSYSEgoState) -> EgoState
        dyn_obj = DynamicObject.deserialize(pubsubMsg.s_DynamicObject)
        return cls(dyn_obj.obj_id, dyn_obj.timestamp
                   , dyn_obj._cached_cartesian_state
                   , dyn_obj._cached_map_state
                   , dyn_obj.size
                   , dyn_obj.confidence, dyn_obj.off_map)


class State(PUBSUB_MSG_IMPL):
    """ Members annotations for python 2 compliant classes """
    occupancy_state = OccupancyState
    dynamic_objects = List[DynamicObject]
    ego_state = EgoState
    is_sampled = bool

    def __init__(self, is_sampled, occupancy_state, dynamic_objects, ego_state):
        # type: (bool, OccupancyState, List[DynamicObject], EgoState) -> None
        """
        main class for the world state. deep copy is required by self.clone_with!
        :param is_sampled: indicates if this state is sampled from a trajectory (and wasn't received from state_module)
        :param occupancy_state: free space
        :param dynamic_objects:
        :param ego_state:
        """
        self.is_sampled = is_sampled
        self.occupancy_state = occupancy_state
        self.dynamic_objects = dynamic_objects
        self.ego_state = ego_state

    def clone_with(self, is_sampled=None, occupancy_state=None, dynamic_objects=None, ego_state=None):
        # type: (bool, OccupancyState, List[DynamicObject], EgoState) -> State
        """
        clones state object with potential overriding of specific fields.
        requires deep-copying of all fields in State.__init__ !!
        """
        return State(is_sampled if is_sampled is not None else self.is_sampled,
                     occupancy_state or self.occupancy_state,
                     dynamic_objects if dynamic_objects is not None else self.dynamic_objects,
                     ego_state or self.ego_state)

    def serialize(self):
        # type: () -> TsSYSState
        pubsub_msg = TsSYSState()

        pubsub_msg.e_b_isSampled = self.is_sampled
        pubsub_msg.s_OccupancyState = self.occupancy_state.serialize()
        ''' resize the list at once to the right length '''
        pubsub_msg.e_Cnt_NumOfObjects = len(self.dynamic_objects)

        for i in range(pubsub_msg.e_Cnt_NumOfObjects):
            pubsub_msg.s_DynamicObjects[i] = self.dynamic_objects[i].serialize()
        pubsub_msg.s_EgoState = self.ego_state.serialize()
        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg):
        # type: (TsSYSState) -> State
        dynamic_objects = list()
        for i in range(pubsubMsg.e_Cnt_NumOfObjects):
            dynamic_objects.append(DynamicObject.deserialize(pubsubMsg.s_DynamicObjects[i]))
        return cls(pubsubMsg.e_b_isSampled
                   , OccupancyState.deserialize(pubsubMsg.s_OccupancyState)
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
