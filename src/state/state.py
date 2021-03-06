import copy
from logging import Logger
from typing import List, Optional, Dict, Tuple, TypeVar

import numpy as np
from decision_making.src.exceptions import MultipleObjectsWithRequestedID, EgoStateLaneIdNotValid
from decision_making.src.global_constants import EGO_LENGTH, EGO_WIDTH, EGO_HEIGHT, LANE_END_COST_IND
from decision_making.src.messages.scene_dynamic_message import SceneDynamic, ObjectLocalization
from decision_making.src.messages.scene_static_enums import ManeuverType
from decision_making.src.messages.scene_static_message import TrafficControlBar
from decision_making.src.messages.serialization import PUBSUB_MSG_IMPL
from decision_making.src.messages.turn_signal_message import TurnSignal
from decision_making.src.planning.behavioral.state.driver_initiated_motion_state import DriverInitiatedMotionState
from decision_making.src.planning.types import C_X, C_Y, C_V, C_YAW, CartesianExtendedState, C_A, C_K, FS_SV, FS_SA
from decision_making.src.planning.types import LaneSegmentID, LaneOccupancyCost, LaneEndCost
from decision_making.src.planning.utils.math_utils import Math
from decision_making.src.state.map_state import MapState
from decision_making.src.utils.map_utils import MapUtils


class DynamicObjectsData:
    def __init__(self, num_objects: int, objects_localization: List[ObjectLocalization], timestamp: int):
        self.num_objects = num_objects
        self.objects_localization = objects_localization
        self.timestamp = timestamp


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


class ObjectSize(PUBSUB_MSG_IMPL):
    length = float
    width = float
    height = float

    def __init__(self, length: float, width: float, height: float) -> None:
        self.length = length
        self.width = width
        self.height = height


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
    is_ghost = bool
    turn_signal = TurnSignal

    def __init__(self, obj_id: int, timestamp: int, cartesian_state: Optional[CartesianExtendedState],
                 map_state: Optional[MapState], size: ObjectSize, confidence: float, off_map: bool,
                 is_ghost: Optional[bool] = False, turn_signal: Optional[TurnSignal] = None):
        """
        Data object that hold
        :param obj_id: object id
        :param timestamp: time of perception [nanosec.]
        :param cartesian_state: localization relative to map's cartesian origin frame
        :param map_state: localization in a map-object's frame (road,segment,lane)
        :param size: class ObjectSize
        :param confidence: of object's existence
        :param off_map: indicates if the vehicle is off the map
        :param is_ghost: indicates whether the object is a projection of a real object in a different lane
        :param turn_signal: turn signal status
        """
        self.obj_id = obj_id
        self.timestamp = timestamp
        self._cached_cartesian_state = cartesian_state
        self._cached_map_state = map_state
        self.size = copy.copy(size)
        self.confidence = confidence
        self.off_map = off_map
        self.is_ghost = is_ghost
        self.turn_signal = turn_signal

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
        :return: [rear left, front left, front right, rear right]

        [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        """
        width = self.size.width / 2.0
        length = self.size.length / 2.0
        cos, sin = np.cos(self.yaw), np.sin(self.yaw)

        return np.array([self.x, self.y]) + np.dot(np.array([
            [[-cos, -sin], [-sin, cos]],
            [[cos, -sin], [sin, cos]],
            [[cos, sin], [sin, -cos]],
            [[-cos, sin], [-sin, -cos]]]),
            np.array([length, width]))

    @property
    def map_state(self):
        # type: () -> MapState
        if self._cached_map_state is None:
            closest_lane_id = MapUtils.get_closest_lane(self.cartesian_state[:(C_Y+1)])
            lane_frenet = MapUtils.get_lane_frenet_frame(closest_lane_id)
            self._cached_map_state = MapState(lane_frenet.cstate_to_fstate(self.cartesian_state), closest_lane_id)
        return self._cached_map_state

    @property
    def timestamp_in_sec(self):
        return Math.ticks_to_sec(self.timestamp)

    @classmethod
    def create_from_cartesian_state(cls, obj_id, timestamp, cartesian_state, size, confidence, off_map, turn_signal = None):
        # type: (int, int, CartesianExtendedState, ObjectSize, float, bool, Optional[TurnSignal]) -> DynamicObject
        """
        Constructor that gets only cartesian-state (without map-state)
        :param obj_id: object id
        :param timestamp: time of perception [nanosec.]
        :param cartesian_state: localization relative to map's cartesian origin frame
        :param size: class ObjectSize
        :param confidence: of object's existence
        :param off_map: indicates if the vehicle is off the map
        :param turn_signal: turn signal status

        """
        return cls(obj_id, timestamp, cartesian_state, None, size, confidence, off_map, turn_signal)

    @classmethod
    def create_from_map_state(cls, obj_id, timestamp, map_state, size, confidence, off_map, turn_signal = None):
        # type: (int, int, MapState, ObjectSize, float, bool, Optional[TurnSignal]) -> DynamicObject
        """
        Constructor that gets only map-state (without cartesian-state)
        :param obj_id: object id
        :param timestamp: time of perception [nanosec.]
        :param map_state: localization in a map-object's frame (road,segment,lane)
        :param size: class ObjectSize
        :param confidence: of object's existence
        :param off_map: is the vehicle is off the map
        :param turn_signal: turn signal status
        """
        return cls(obj_id, timestamp, None, map_state, size, confidence, off_map, turn_signal)

    def clone_from_cartesian_state(self, cartesian_state, timestamp_in_sec=None):
        # type: (CartesianExtendedState, Optional[float]) -> DynamicObject
        """clones self while overriding cartesian_state and optionally timestamp"""
        return self.__class__.create_from_cartesian_state(self.obj_id,
                                                          Math.sec_to_ticks(timestamp_in_sec or self.timestamp_in_sec),
                                                          cartesian_state,
                                                          self.size, self.confidence, self.off_map, self.turn_signal)

    def clone_from_map_state(self, map_state, timestamp_in_sec=None):
        # type: (MapState, Optional[float]) -> DynamicObject
        """clones self while overriding map_state and optionally timestamp"""
        return self.create_from_map_state(self.obj_id,
                                          Math.sec_to_ticks(timestamp_in_sec or self.timestamp_in_sec),
                                          map_state,
                                          self.size, self.confidence, self.off_map, self.turn_signal)


class EgoState(DynamicObject):
    _dim_state = DriverInitiatedMotionState

    def __init__(self, obj_id: int, timestamp: int, cartesian_state: CartesianExtendedState, map_state: MapState,
                 size: ObjectSize, confidence: float, off_map: bool, turn_signal: Optional[TurnSignal] = None,
                 dim_state: DriverInitiatedMotionState = None):
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
        :param turn_signal: turn signal status
        :param dim_state: driver initiated motion state
        """
        super(self.__class__, self).__init__(obj_id=obj_id, timestamp=timestamp, cartesian_state=cartesian_state,
                                             map_state=map_state, size=size, confidence=confidence, off_map=off_map,
                                             turn_signal=turn_signal)
        self._dim_state = dim_state

    def update_dim_state(self, ego_s: float, closestTCB: Tuple[TrafficControlBar, float], ignored_TCB_distance: float) -> None:
        """
        Update DIM state machine of ego
        :param ego_s: s location of ego in GeneralizedFrenetSerretFrame
        :param closestTCB: Tuple of TCB object and its s location in GeneralizedFrenetSerretFrame
        :param ignored_TCB_distance: ignored s location
        """
        if self._dim_state is not None:
            self._dim_state.update_state(self.timestamp_in_sec, self._cached_map_state.lane_fstate, ego_s, closestTCB,
                                         ignored_TCB_distance)

    def get_stop_bar_to_ignore(self):
        """
        Used by driver initiated motion: return stop bar id to ignore if acceleration pedal was pressed
        :return: stop bar id that should be ignored
        """
        return self._dim_state.stop_bar_to_ignore() if self._dim_state is not None else None


T = TypeVar('T', bound='State')


class State(PUBSUB_MSG_IMPL):
    """ Members annotations for python 2 compliant classes """
    occupancy_state = OccupancyState
    dynamic_objects = List[DynamicObject]
    ego_state = EgoState
    is_sampled = bool

    def __init__(self, is_sampled: bool, occupancy_state: OccupancyState,
                 dynamic_objects: List[DynamicObject], ego_state: EgoState) -> None:
        """
        main class for the world state. deep copy is required by self.clone_with!
        :param is_sampled: indicates if this state is sampled from a trajectory (and wasn't received from scene_dynamic)
        :param occupancy_state: free space
        :param dynamic_objects:
        :param ego_state:
        """
        self.is_sampled = is_sampled
        self.occupancy_state = occupancy_state
        self.dynamic_objects = dynamic_objects
        self.ego_state = ego_state

    def clone_with(self, is_sampled: bool = None, occupancy_state: OccupancyState = None,
                   dynamic_objects: List[DynamicObject] = None, ego_state: EgoState = None) -> T:
        """
        clones state object with potential overriding of specific fields.
        requires deep-copying of all fields in State.__init__ !!
        """
        return State(is_sampled if is_sampled is not None else self.is_sampled,
                     occupancy_state or self.occupancy_state,
                     dynamic_objects if dynamic_objects is not None else self.dynamic_objects,
                     ego_state or self.ego_state)

    # TODO: remove when access to dynamic objects according to dictionary will be available.
    @classmethod
    def get_object_from_state(cls, state: T, target_obj_id: int) -> DynamicObject:
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
    def get_objects_from_state(cls, state: T, target_obj_ids: List[int]) -> List[DynamicObject]:
        """
        Returns a list of object with the specific obj_ids from state
        :param state: the state to query
        :param target_obj_ids: a list of the id of the requested objects
        :return: the dynamic_objects matching the requested ids
        """
        selected_objects = [obj for obj in state.dynamic_objects if obj.obj_id in target_obj_ids]
        return selected_objects

    def handle_negative_velocities(self, logger: Logger) -> None:
        """
        Handles cases of ego state or dynamic objects with negative velocities.
        It modifies their velocities so they will equal zero and writes a warning in the log.
        :param logger: Logging module
        :return: None
        """
        if self.ego_state.cartesian_state[C_V] < 0:
            logger.warning('Ego was received with negative velocity %f' % self.ego_state.cartesian_state[C_V])
            self.ego_state.cartesian_state[C_V] = 0
            self.ego_state.map_state.lane_fstate[FS_SV] = 0
        elif self.ego_state.cartesian_state[C_V] == 0 and self.ego_state.cartesian_state[C_A] < 0:
            logger.warning('Ego was received with zero velocity and negative acceleration %f'
                           % self.ego_state.cartesian_state[C_A])
            self.ego_state.cartesian_state[C_A] = 0
            self.ego_state.map_state.lane_fstate[FS_SA] = 0

        for i in range(len(self.dynamic_objects)):
            if self.dynamic_objects[i].cartesian_state[C_V] < 0:
                logger.warning(
                    'Dynamic object with obj_id %s was received with negative velocity %f',
                    self.dynamic_objects[i].obj_id, self.dynamic_objects[i].cartesian_state[C_V])
                self.dynamic_objects[i].cartesian_state[C_V] = 0
                self.dynamic_objects[i].map_state.lane_fstate[FS_SV] = 0

    @classmethod
    def create_state_from_scene_dynamic(cls, scene_dynamic: SceneDynamic,
                                        selected_gff_segment_ids: np.ndarray,
                                        logger: Logger,
                                        route_plan_dict: Optional[Dict[LaneSegmentID, Tuple[LaneOccupancyCost, LaneEndCost]]] = None,
                                        turn_signal: Optional[TurnSignal] = None,
                                        dim_state: DriverInitiatedMotionState = None):
        """
        This methods takes an already deserialized SceneDynamic message and converts it to a State object
        :param scene_dynamic: scene dynamic data
        :param selected_gff_segment_ids: list of GFF segment ids for last or current BP action. If this method is called
               from inside BP, the parameter refers to the last chosen BP action, while it refers to current BP action
               if called from inside TP
        :param logger: Logging module
        :param route_plan_dict: dictionary data with lane id as key and tuple of (occupancy and end costs) as value.
               Note that it is an optional argument and is used only when the method is called from inside BP and the
               previous BP action is not available, e.g., at the beginning of the planning time.
        :param turn_signal: turn signal status
        :param dim_state: driver initiated motion state
        :return: valid State class
        """

        timestamp = Math.sec_to_ticks(scene_dynamic.s_Data.s_RecvTimestamp.timestamp_in_seconds)
        occupancy_state = OccupancyState(0, np.array([0]), np.array([0]))

        selected_host_hyp_idx = State.select_ego_hypothesis(scene_dynamic, selected_gff_segment_ids, logger, route_plan_dict)

        ego_map_state = MapState(lane_fstate=scene_dynamic.s_Data.s_host_localization.
                                 as_host_hypothesis[selected_host_hyp_idx].a_lane_frenet_pose,
                                 lane_id=scene_dynamic.s_Data.s_host_localization.
                                 as_host_hypothesis[selected_host_hyp_idx].e_i_lane_segment_id)

        ego_state = EgoState(obj_id=0,
                             timestamp=timestamp,
                             cartesian_state=scene_dynamic.s_Data.s_host_localization.a_cartesian_pose,
                             map_state=ego_map_state,
                             size=ObjectSize(EGO_LENGTH, EGO_WIDTH, EGO_HEIGHT),
                             confidence=1.0, off_map=False, turn_signal=turn_signal,  dim_state=dim_state)

        dyn_obj_data = DynamicObjectsData(num_objects=scene_dynamic.s_Data.e_Cnt_num_objects,
                                          objects_localization=scene_dynamic.s_Data.as_object_localization,
                                          timestamp=timestamp)

        dynamic_objects = State.create_dyn_obj_list(dyn_obj_data)

        return cls(False, occupancy_state, dynamic_objects, ego_state)

    @staticmethod
    def select_ego_hypothesis(scene_dynamic: SceneDynamic,
                              selected_gff_segment_ids: np.ndarray,
                              logger: Logger,
                              route_plan_dict: Optional[Dict[LaneSegmentID, Tuple[LaneOccupancyCost, LaneEndCost]]]) -> int:
        """
        selects the correct ego localization hypothesis among possible multiple hypotheses published by scene_dynamic
        :param scene_dynamic: scene dynamic data
        :param selected_gff_segment_ids: list of GFF segment ids for last or current BP action. If this method is called
               from inside BP, the parameter refers to the last chosen BP action, while it refers to current BP action
               if called from inside TP
        :param logger: Logging module
        :param route_plan_dict: dictionary data with lane id as key and tuple of (occupancy and end costs) as value.
               Note that it is an optional argument and is used only when the method is called from inside BP and the
               previous BP action is not available, e.g., at the beginning of the planning time.
        :return: index for the selected localization hypothesis.
        """

        selected_host_hyp_idx = 0

        host_hyp_lane_ids = [hyp.e_i_lane_segment_id
                             for hyp in scene_dynamic.s_Data.s_host_localization.as_host_hypothesis]

        if len(host_hyp_lane_ids) > 1 and len(selected_gff_segment_ids) > 0:

            # find all common lane indices in host hypotheses and last gff segments
            # take the hyp. whose lane has the least distance from the host, i.e.,
            # the min. index in host_hyp_lane_ids since it is sorted based on the distance
            common_lanes_indices = np.where(np.isin(host_hyp_lane_ids, selected_gff_segment_ids))[0]

            if len(common_lanes_indices) > 0:
                selected_host_hyp_idx = common_lanes_indices[0]
            else:
                # there are no common ids between localization and prev. gff
                # raise a warning and choose the closet lane
                logger.warning("None of the host localization hypotheses matches the previous planning action")

        elif len(host_hyp_lane_ids) > 1 and route_plan_dict is not None:
            # If previous action does not exist, choose the hypothesis lane with minimum route plan end cost
            # if there are multiple lanes with similar minimum cost values, the lane with STRAIGHT maneuver type is chosen
            # if there is no lane with STRAIGHT maneuver type or multiple lanes have STRAIGHT maneuver type (in lane change scenario),
            # the closest lane (smallest index) is chosen
            try:
                lane_end_costs = [route_plan_dict[lane_id][LANE_END_COST_IND] for lane_id in host_hyp_lane_ids]
                min_indices = np.argwhere(lane_end_costs == np.min(lane_end_costs))
                selected_host_hyp_idx = min_indices[0][0]
                if len(min_indices) > 1:
                    maneuver_types = [MapUtils.get_lane_maneuver_type(host_hyp_lane_ids[idx[0]]) for idx in min_indices]
                    if ManeuverType.STRAIGHT_CONNECTION in maneuver_types:
                        selected_host_hyp_idx = [min_indices[idx][0] for idx in range(len(maneuver_types))
                                                 if maneuver_types[idx] == ManeuverType.STRAIGHT_CONNECTION][0]
            except KeyError:
                # if route plan cost is not found for any host lanes, raise a warning and continue with the closest lane
                logger.warning("Route plan cost not found for a host lane segment")

        if host_hyp_lane_ids[selected_host_hyp_idx] == 0:
            raise EgoStateLaneIdNotValid("Ego vehicle lane assignment is not valid")

        return selected_host_hyp_idx

    @staticmethod
    def create_dyn_obj_list(dyn_obj_data: DynamicObjectsData) -> List[DynamicObject]:
        """
        Convert serialized object perception and global localization data into a DM object (This also includes
        computation of the object's road localization). Additionally store the object in memory as preparation for
        the case where it will leave the field of view.
        :param dyn_obj_data:
        :return: List of dynamic object in DM format.
        """
        timestamp = dyn_obj_data.timestamp
        objects_list = []
        for obj_idx in range(dyn_obj_data.num_objects):
            obj_loc = dyn_obj_data.objects_localization[obj_idx]
            obj_id = obj_loc.e_Cnt_object_id
            cartesian_state = obj_loc.a_cartesian_pose
            # TODO: Handle multiple hypotheses
            map_state = MapState(obj_loc.as_object_hypothesis[0].a_lane_frenet_pose, obj_loc.as_object_hypothesis[0].e_i_lane_segment_id)
            size = ObjectSize(obj_loc.s_bounding_box.e_l_length,
                              obj_loc.s_bounding_box.e_l_width,
                              obj_loc.s_bounding_box.e_l_height)
            confidence = obj_loc.as_object_hypothesis[0].e_r_probability
            off_map = obj_loc.as_object_hypothesis[0].e_b_off_lane
            dyn_obj = DynamicObject(obj_id=obj_id,
                                    timestamp=timestamp,
                                    cartesian_state=cartesian_state,
                                    map_state=map_state if map_state.lane_id > 0 else None,
                                    size=size,
                                    confidence=confidence,
                                    off_map=off_map)

            objects_list.append(dyn_obj)  # update the list of dynamic objects

        return objects_list
