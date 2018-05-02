from enum import Enum
from typing import Tuple, List
import numpy as np
import copy
from decision_making.src.planning.trajectory.trajectory_planner import SamplableTrajectory
from decision_making.src.planning.types import FS_DX, FS_SX, FS_DV, FS_SV, FS_SA, C_X, C_Y, C_YAW, C_V, C_K
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.state.state import DynamicObject, State, ObjectSize, EgoState
from mapping.src.model.localization import RoadLocalization


class RelativeLane(Enum):
    """"
    The lane associated with a certain Recipe, relative to ego
    """
    RIGHT_LANE = -1
    SAME_LANE = 0
    LEFT_LANE = 1


class RelativeLongitudinalPosition(Enum):
    """"
    The high-level longitudinal position associated with a certain Recipe, relative to ego
    """
    REAR = -1
    PARALLEL = 0
    FRONT = 1


class ActionType(Enum):
    """"
    Type of Recipe, when "follow lane" is a static action while "follow vehicle" and "takeover vehicle" are dynamic ones.
    """
    FOLLOW_LANE = 1
    FOLLOW_VEHICLE = 2
    TAKE_OVER_VEHICLE = 3


class AggressivenessLevel(Enum):
    """"
    Aggressiveness driving level, which affects the urgency in reaching the specified goal.
    """
    CALM = 0
    STANDARD = 1
    AGGRESSIVE = 2


# Define semantic cell
SemanticGridCell = Tuple[int, int]

# tuple indices
LAT_CELL, LON_CELL = 0, 1


class ActionRecipe:
    def __init__(self, relative_lane: RelativeLane, action_type: ActionType, aggressiveness: AggressivenessLevel):
        self.relative_lane = relative_lane
        self.action_type = action_type
        self.aggressiveness = aggressiveness

    @classmethod
    def from_args_list(cls, args: List):
        return cls(*args)


class StaticActionRecipe(ActionRecipe):
    """"
    Data object containing the fields needed for specifying a certain static action, together with the state.
    """
    def __init__(self, relative_lane: RelativeLane, velocity: float, aggressiveness: AggressivenessLevel):
        super().__init__(relative_lane, ActionType.FOLLOW_LANE, aggressiveness)
        self.velocity = velocity


class DynamicActionRecipe(ActionRecipe):
    """"
    Data object containing the fields needed for specifying a certain dynamic action, together with the state.
    """
    def __init__(self, relative_lane: RelativeLane, relative_lon: RelativeLongitudinalPosition,  action_type: ActionType, aggressiveness: AggressivenessLevel):
        super().__init__(relative_lane, action_type, aggressiveness)
        self.relative_lon = relative_lon


class ActionSpec:
    """
    Holds the actual translation of the semantic action in terms of trajectory specifications.
    """

    def __init__(self, t: float, v: float, s: float, d: float, samplable_trajectory: SamplableTrajectory = None):
        """
        The trajectory specifications are defined by the target ego state
        :param t: time [sec]
        :param v: velocity [m/s]
        :param s: relative longitudinal distance to ego in Frenet frame [m]
        :param d: relative lateral distance to ego in Frenet frame [m]
        :param samplable_trajectory: samplable reference trajectory.
        """
        self.t = t
        self.v = v
        self.s = s
        self.d = d
        self.samplable_trajectory = samplable_trajectory

    def __str__(self):
        return str({k: str(v) for (k, v) in self.__dict__.items()})


class NavigationGoal:
    def __init__(self, road_id: int, lon: float, from_lane: int, to_lane: int):
        self.road_id = road_id
        self.lon = lon
        self.from_lane = from_lane
        self.to_lane = to_lane


class FrenetObject:
    # def __init__(self, dynamic_object: DynamicObject):
    #     self.timestamp = dynamic_object.timestamp
    #     self.road_localization = dynamic_object.road_localization
    #     self.size = dynamic_object.size
    #     road_yaw = dynamic_object.road_localization.intra_road_yaw
    #     self.fstate = np.array([dynamic_object.road_localization.road_lon,
    #                             dynamic_object.road_longitudinal_speed,
    #                             dynamic_object.acceleration_lon * np.cos(road_yaw),
    #                             dynamic_object.road_localization.intra_road_lat,
    #                             dynamic_object.road_lateral_speed,
    #                             dynamic_object.acceleration_lon * np.sin(road_yaw)])

    def __init__(self, timestamp: int, fstate: np.array, size: ObjectSize, road_id: int, lane_width: float):
        self.timestamp = timestamp
        (lon, lat) = (fstate[FS_SX], fstate[FS_DX])
        self.road_localization = RoadLocalization(road_id, int(lat/lane_width), lat, lat % lane_width, lon,
                                                  np.arctan2(fstate[FS_DV], fstate[FS_SV]))
        self.size = copy.deepcopy(size)
        self.fstate = copy.deepcopy(fstate)

    def create_dynamic_object(self, obj_id, frenet_frame: FrenetSerret2DFrame) -> DynamicObject:
        cstate = frenet_frame.fstate_to_cstate(self.fstate)
        return DynamicObject(obj_id, self.timestamp, cstate[C_X], cstate[C_Y], 0, cstate[C_YAW],
                             self.size, 0, cstate[C_V], 0, self.fstate[FS_SA], cstate[C_K]*cstate[C_V])

    def create_ego_state(self, frenet_frame: FrenetSerret2DFrame) -> EgoState:
        obj = self.create_dynamic_object(0, frenet_frame)
        return EgoState(obj.obj_id, obj.timestamp, obj.x, obj.y, obj.z, obj.yaw, obj.size, obj.confidence,
                        obj.v_x, obj.v_y, obj.acceleration_lon, obj.omega_yaw, 0)


class FrenetState:
    def __init__(self, ego_state: FrenetObject, dynamic_objects: List[FrenetObject]):
        self.ego_state = ego_state
        self.dynamic_objects = dynamic_objects

    def create_state(self, frenet_frame: FrenetSerret2DFrame) -> State:
        dynamic_objects = []
        for i, obj in enumerate(self.dynamic_objects):
            dynamic_objects.append(obj.create_dynamic_object(i+1, frenet_frame))
        return State(None, dynamic_objects, self.ego_state.create_ego_state(frenet_frame))
