from enum import Enum
from typing import List
import numpy as np

from decision_making.src.planning.types import FrenetState2D


class ActionType(Enum):
    """"
    Type of Recipe, when "follow lane" is a static action while "follow vehicle" and "takeover vehicle" are dynamic ones.
    """
    FOLLOW_LANE = 1
    FOLLOW_VEHICLE = 2
    OVERTAKE_VEHICLE = 3


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
        return 'DynamicActionRecipe: %s' % self.__dict__


class DynamicActionRecipe(ActionRecipe):
    """"
    Data object containing the fields needed for specifying a certain dynamic action, together with the state.
    """
    def __init__(self, relative_lane: RelativeLane, relative_lon: RelativeLongitudinalPosition, action_type: ActionType,
                 aggressiveness: AggressivenessLevel):
        super().__init__(relative_lane, action_type, aggressiveness)
        self.relative_lon = relative_lon

    def __str__(self):
        return 'DynamicActionRecipe: %s' % self.__dict__


class ActionSpec:
    """
    Holds the actual translation of the semantic action in terms of trajectory specifications.
    """
    def __init__(self, t: float, v: float, s: float, d: float, relative_lane: RelativeLane):
        """
        The trajectory specifications are defined by the target ego state
        :param t: time [sec]
        :param v: velocity [m/s]
        :param s: global longitudinal position in Frenet frame [m]
        :param d: global lateral position in Frenet frame [m]
        :param relative_lane: relative target lane
        """
        self.t = t
        self.v = v
        self.s = s
        self.d = d
        self.relative_lane = relative_lane

    def __str__(self):
        return str({k: str(v) for (k, v) in self.__dict__.items()})

    def as_fstate(self) -> FrenetState2D:
        return np.array([self.s, self.v, 0, self.d, 0, 0])
