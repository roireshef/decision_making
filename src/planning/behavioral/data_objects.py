from enum import Enum
from typing import List
import numpy as np

from decision_making.src.global_constants import TRAJECTORY_TIME_RESOLUTION
from decision_making.src.planning.types import FrenetState2D


class ActionType(Enum):
    """"
    Type of Recipe, when "follow lane" is a static action while "follow vehicle" and "takeover vehicle" and "follow_road_sign" are dynamic ones.
    """
    FOLLOW_LANE = 1
    FOLLOW_VEHICLE = 2
    OVERTAKE_VEHICLE = 3
    FOLLOW_ROAD_SIGN = 4


class AggressivenessLevel(Enum):
    """"
    Aggressiveness driving level, which affects the urgency in reaching the specified goal.
    """
    CALM = 0
    STANDARD = 1
    AGGRESSIVE = 2


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

    def __str__(self):
        return 'StaticActionRecipe: %s' % self.__dict__


class TargetActionRecipe(ActionRecipe):
    """"
    Data object containing the fields needed for specifying a certain target related action, together with the state.
    Currently the supported targets are vehicles and road signs
    """
    def __init__(self, relative_lane: RelativeLane, relative_lon: RelativeLongitudinalPosition, action_type: ActionType,
                 aggressiveness: AggressivenessLevel):
        super().__init__(relative_lane, action_type, aggressiveness)
        self.relative_lon = relative_lon

    def __str__(self):
        return 'TargetActionRecipe: %s' % self.__dict__


class DynamicActionRecipe(TargetActionRecipe):
    """"
    Data object containing the fields needed for specifying a certain dynamic action, together with the state.
    Dynamic actions are actions defined with respect to target vehicles
    """
    def __init__(self, relative_lane: RelativeLane, relative_lon: RelativeLongitudinalPosition, action_type: ActionType,
                 aggressiveness: AggressivenessLevel):
        super().__init__(relative_lane, relative_lon, action_type, aggressiveness)

    def __str__(self):
        return 'DynamicActionRecipe: %s' % self.__dict__


class RoadSignActionRecipe(TargetActionRecipe):
    """"
    Data object containing the fields needed for specifying a certain road sign action, together with the state.
    """
    def __init__(self, relative_lane: RelativeLane, relative_lon: RelativeLongitudinalPosition, action_type: ActionType,
                 aggressiveness: AggressivenessLevel):
        super().__init__(relative_lane, relative_lon, action_type, aggressiveness)

    def __str__(self):
        return 'RoadSignActionRecipe: %s' % self.__dict__


class ActionSpec:
    """
    Holds the actual translation of the semantic action in terms of trajectory specifications.
    """
    def __init__(self, t: float, v: float, s: float, d: float, recipe: ActionRecipe):
        """
        The trajectory specifications are defined by the target ego state
        :param t: time [sec]
        :param v: velocity [m/s]
        :param s: global longitudinal position in Frenet frame [m]
        :param d: global lateral position in Frenet frame [m]
        :param recipe: the original recipe that the action space originated from (redundant but stored for efficiency)
        """
        self.t = t
        self.v = v
        self.s = s
        self.d = d
        self.recipe = recipe

    @property
    def relative_lane(self):
        return self.recipe.relative_lane

    @property
    def only_padding_mode(self):
        """ if planning time is shorter than the TP's time resolution, the result will be only padding in the TP"""
        return self.t < TRAJECTORY_TIME_RESOLUTION

    def __str__(self):
        return str({k: str(v) for (k, v) in self.__dict__.items()})

    def as_fstate(self) -> FrenetState2D:
        return np.array([self.s, self.v, 0, self.dx, self.dv, self.da])
