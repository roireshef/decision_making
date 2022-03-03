from enum import Enum
from typing import Optional, NamedTuple

import numpy as np

from decision_making.src.global_constants import TRAJECTORY_TIME_RESOLUTION
from decision_making.src.planning.behavioral.data_objects import AggressivenessLevel
from decision_making.src.planning.types import FrenetState2D
from decision_making.src.rl_agent.environments.uc_rl_map import RelativeLane
from decision_making.src.rl_agent.global_types import LateralDirection
from decision_making.src.rl_agent.utils.class_utils import Representable, CloneableNamedTuple
from decision_making.src.rl_agent.utils.samplable_werling_trajectory import \
    SamplableWerlingTrajectory


class LaneChangeEvent(Enum):
    UNCHANGED = 0
    NEGOTIATE = 1
    COMMIT = 2
    ABORT = 3


class RLActionRecipe(Representable):
    def __init__(self, lateral_dir: LateralDirection):
        self.lateral_dir = lateral_dir

    @property
    def is_noop(self) -> bool:
        return False


class TerminalVelocityRLActionRecipe(RLActionRecipe):
    def __init__(self, lateral_dir: LateralDirection, velocity: float, aggressiveness: AggressivenessLevel):
        """
        An abstract class for storing recipe information.
        :param lateral_dir: lateral direction of the action - intentionally kept abstract since its meaning changes
        with the context (different action spaces use this value differently)
        :param velocity: target longitudinal velocity
        :param aggressiveness: aggressiveness level that determines longitudinal horizon
        """
        super().__init__(lateral_dir)
        self.velocity = velocity
        self.aggressiveness = aggressiveness

    def __str__(self):
        return 'RLActionRecipe: %s' % self.__dict__

    def __lt__(self, other):
        return hash(self) < hash(other)


class KeepLaneTerminalVelocityActionRecipe(TerminalVelocityRLActionRecipe):
    def __init__(self, velocity: float, aggressiveness: AggressivenessLevel):
        super().__init__(LateralDirection.SAME, velocity, aggressiveness)


class CommitLaneChangeTerminalVelocityActionRecipe(TerminalVelocityRLActionRecipe):
    """ Here, lane change actions are commitments to execute X consecutive times for the same direction of
    <lateral_dir> """
    pass


class LateralOffsetTerminalVelocityActionRecipe(TerminalVelocityRLActionRecipe):
    """ Here, lateral_dir represents the direction to move the target lateral offset to """
    pass


class ConstantAccelerationRLActionRecipe(RLActionRecipe):
    def __init__(self, lateral_dir: LateralDirection, acceleration: float):
        """
        Action recipe that applies constant acceleration for the duration of a single timestamp (longitudinally)
        :param lateral_dir: lateral direction of the action - intentionally kept abstract since its meaning changes
        with the context (different action spaces use this value differently)
        :param acceleration: the acceleration to apply
        """
        super().__init__(lateral_dir)
        self.acceleration = acceleration

    @property
    def is_noop(self) -> bool:
        return self.lateral_dir == LateralDirection.SAME and self.acceleration == 0

    def __str__(self):
        return 'ConstantAccelerationActionRecipe: %s' % self.__dict__

    def __lt__(self, other):
        return hash(self) < hash(other)


class RLActionSpec(NamedTuple, CloneableNamedTuple):
    """
    The trajectory specifications are defined by the target ego state
    Holds the actual translation of the semantic action in terms of trajectory specifications.
    :param t: time [sec]
    :param v: velocity [m/s]
    :param s: global longitudinal position in Frenet frame [m]
    :param d: global lateral position in Frenet frame [m]
    :param recipe: the original recipe that the action space originated from (redundant but stored for efficiency)
    """
    t: float
    v: float
    s: float
    d: float
    relative_lane: RelativeLane
    recipe: RLActionRecipe
    baseline_trajectory: Optional[SamplableWerlingTrajectory] = None
    lane_change_event: LaneChangeEvent = LaneChangeEvent.UNCHANGED

    @property
    def only_padding_mode(self):
        """ if planning time is shorter than the TP's time resolution, the result will be only padding in the TP"""
        return self.is_only_padding_mode(self.t)

    def __str__(self):
        return str({k: str(v) for (k, v) in self.__dict__.items()})

    def as_fstate(self) -> FrenetState2D:
        return np.array([self.s, self.v, 0, self.d, 0, 0])

    @staticmethod
    def is_only_padding_mode(time: float):
        return time < TRAJECTORY_TIME_RESOLUTION
