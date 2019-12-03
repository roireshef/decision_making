from enum import Enum
from typing import List
import numpy as np

from decision_making.src.global_constants import TRAJECTORY_TIME_RESOLUTION, MIN_OFFSET_FOR_LANE_CHANGE_COMPLETE, \
    MIN_REL_HEADING_FOR_LANE_CHANGE_COMPLETE
from decision_making.src.planning.types import FrenetState2D, FS_DX, FS_SX, C_YAW
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GFFType, GeneralizedFrenetSerretFrame
from decision_making.src.messages.turn_signal_message import TurnSignalState

# This is done to allow type checking without circular imports at runtime
# the TYPE_CHECKING variable is always false at runtime, avoiding circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from decision_making.src.planning.behavioral.state.behavioral_grid_state import BehavioralGridState


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
        return np.array([self.s, self.v, 0, self.d, 0, 0])


class LaneChangeInfo:
    def __init__(self, source_lane_gff: GeneralizedFrenetSerretFrame, target_lane_ids: np.ndarray,
                 lane_change_active: bool, lane_change_start_time: float):
        """
        Holds lane change information
        :param source_lane_gff: GFF that the host was in when a lane change was initiated
        :param target_lane_ids: Lane IDs of the GFF that the host is targeting in a lane change
        :param lane_change_active: True when a lane change is active; otherwise, False
        :param lane_change_start_time: Time when a lane change began
        """
        self.source_lane_gff = source_lane_gff
        self._target_lane_ids = target_lane_ids
        self.lane_change_active = lane_change_active
        self.lane_change_start_time = lane_change_start_time

    def __str__(self):
        # print as dict for logs
        return str(self.__dict__)

    def _reset(self):
        self.source_lane_gff = None
        self._target_lane_ids = np.array([])
        self.lane_change_active = False
        self.lane_change_start_time = 0.0

    def are_target_lane_ids_in_gff(self, gff: GeneralizedFrenetSerretFrame):
        """
        This checks whether the lane IDs in the target GFF are in the given GFF. This can be used to see whether the host has crossed into the target lane during a lane change.
        :param gff:
        :return: Returns True if any lane IDs in the target GFF are in the given GFF
        """
        return np.any(np.isin(self._target_lane_ids, gff.segment_ids))

    def update(self, behavioral_state: 'BehavioralGridState',  selected_action: ActionSpec):
        """
        Update the lane change status based on the target GFF requested
        :param behavioral_state:
        :param selected_action:
        :return:
        """
        if self.lane_change_active:
            # If lane change is currently active, check to see whether lane change is complete
            is_host_in_target_lane = self.are_target_lane_ids_in_gff(behavioral_state.extended_lane_frames[RelativeLane.SAME_LANE])

            if is_host_in_target_lane:
                distance_to_target_lane_center = behavioral_state.projected_ego_fstates[RelativeLane.SAME_LANE][FS_DX]

                host_station_in_target_lane_gff = np.array([behavioral_state.projected_ego_fstates[RelativeLane.SAME_LANE][FS_SX]])
                relative_heading = behavioral_state.ego_state.cartesian_state[C_YAW] - \
                                   behavioral_state.extended_lane_frames[RelativeLane.SAME_LANE].get_yaw(host_station_in_target_lane_gff)[0]

                # If lane change completion requirements are met and the turn signal has been turned off, the lane change is complete.
                if (abs(distance_to_target_lane_center) < MIN_OFFSET_FOR_LANE_CHANGE_COMPLETE
                        and abs(relative_heading) < MIN_REL_HEADING_FOR_LANE_CHANGE_COMPLETE
                        and behavioral_state.ego_state.turn_signal.s_Data.e_e_turn_signal_state == TurnSignalState.CeSYS_e_Off):
                    self._reset()
        else:
            # If lane change is not currently active, check to see whether a lane change action was selected
            if (selected_action.relative_lane in [RelativeLane.LEFT_LANE, RelativeLane.RIGHT_LANE]
                and behavioral_state.extended_lane_frames[selected_action.relative_lane].gff_type not in
                    [GFFType.Augmented, GFFType.AugmentedPartial]):
                self.lane_change_active = True
                self.source_lane_gff = behavioral_state.extended_lane_frames[RelativeLane.SAME_LANE]
                self._target_lane_ids = behavioral_state.extended_lane_frames[selected_action.relative_lane].segment_ids
                self.lane_change_start_time = behavioral_state.ego_state.timestamp_in_sec
