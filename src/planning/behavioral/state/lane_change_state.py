from enum import Enum
import numpy as np
from typing import Optional, Dict, List

from decision_making.src.global_constants import MAX_OFFSET_FOR_LANE_CHANGE_COMPLETE, MAX_REL_HEADING_FOR_LANE_CHANGE_COMPLETE, \
    LANE_CHANGE_DELAY, LANE_CHANGE_ABORT_THRESHOLD
from decision_making.src.messages.turn_signal_message import TurnSignalState
from decision_making.src.planning.behavioral.data_objects import ActionSpec, RelativeLane
from decision_making.src.planning.types import FS_DX, FS_SX, C_YAW, FrenetState2D
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GFFType, GeneralizedFrenetSerretFrame
from decision_making.src.state.state import EgoState
from decision_making.src.utils.map_utils import MapUtils



class LaneChangeStatus(Enum):
    LaneChangeRequestable = 0   # TODO: Find a better name so that it's not confused with LaneChangeRequested?
    LaneChangeRequested = 1
    AnalyzingSafety = 2
    LaneChangeActiveInSourceLane = 3
    LaneChangeActiveInTargetLane = 4
    LaneChangeCompleteWaitingForReset = 5


class LaneChangeState:
    # This is a class variable and shared by all instances of LaneChangeState
    expected_turn_signal_state = {RelativeLane.LEFT_LANE: TurnSignalState.CeSYS_e_LeftTurnSignalOn,
                                  RelativeLane.RIGHT_LANE: TurnSignalState.CeSYS_e_RightTurnSignalOn}

    def __init__(self, source_lane_gff: Optional[GeneralizedFrenetSerretFrame] = None, target_lane_ids: Optional[np.ndarray] = None,
                 lane_change_start_time: Optional[float] = None, target_relative_lane: Optional[RelativeLane] = None,
                 status: Optional[LaneChangeStatus] = LaneChangeStatus.LaneChangeRequestable):
        """
        Holds lane change state
        :param source_lane_gff: GFF that the host was in when a lane change was initiated
        :param target_lane_ids: Lane IDs of the GFF that the host is targeting in a lane change
        :param lane_change_start_time: Time when a lane change began
        :param target_relative_lane: Relative lane of the target lane during a lane change
        :param status: lane change status
        """
        self.source_lane_gff = source_lane_gff
        self._target_lane_ids = target_lane_ids or np.array([])
        self.lane_change_start_time = lane_change_start_time
        self.target_relative_lane = target_relative_lane
        self.status = status

    def __str__(self):
        # print as dict for logs
        return str(self.__dict__)

    def _reset(self):
        self.source_lane_gff = None
        self._target_lane_ids = np.array([])
        self.lane_change_start_time = None
        self.target_relative_lane = None
        self.status = LaneChangeStatus.LaneChangeRequestable

    def get_target_lane_gff(self, extended_lane_frames: Dict[RelativeLane, GeneralizedFrenetSerretFrame]) -> GeneralizedFrenetSerretFrame:
        """
        Given a set of extended_lane_frames from a BehavioralGridState, this chooses the GFF that represents the
        lane change target lane.
        :param extended_lane_frames:
        :return:
        """
        if self.status in [LaneChangeStatus.AnalyzingSafety, LaneChangeStatus.LaneChangeActiveInSourceLane]:
            target_gff = extended_lane_frames[self.target_relative_lane]
        else:
            target_gff = extended_lane_frames[RelativeLane.SAME_LANE]

        return target_gff

    def _is_lane_id_in_target_lane_ids(self, lane_id: int) -> bool:
        """
        Given a lane ID, this returns True if it's in the target lane
        :param lane_id:
        :return:
        """
        return lane_id in self._target_lane_ids

    def get_lane_change_mask(self, relative_lanes: List[RelativeLane],
                             extended_lane_frames: Dict[RelativeLane, GeneralizedFrenetSerretFrame]) -> List[bool]:
        """
        Given a list of relative lanes from proposed action recipes or specs, this returns a mask for the lane change actions
        :param relative_lanes:
        :param extended_lane_frames:
        :return:
        """
        if self.status in [LaneChangeStatus.AnalyzingSafety, LaneChangeStatus.LaneChangeActiveInSourceLane]:
            return [relative_lane == self.target_relative_lane
                    and extended_lane_frames[relative_lane].gff_type not in [GFFType.Augmented, GFFType.AugmentedPartial]
                    for relative_lane in relative_lanes]
        elif self.status == LaneChangeStatus.LaneChangeActiveInTargetLane:
            return [relative_lane == RelativeLane.SAME_LANE for relative_lane in relative_lanes]
        else:
            return [False] * len(relative_lanes)

    def is_safe_to_start_lane_change(self) -> bool:
        # TODO when safety check is added, should return here the actual safety check result
        return self.status == LaneChangeStatus.AnalyzingSafety

    def update_pre_iteration(self, ego_state: EgoState):
        """
        Updates the lane change status before an iteration
        :param ego_state: state of host
        :return:
        """
        if self.status == LaneChangeStatus.LaneChangeRequestable:
            if ego_state.turn_signal.s_Data.e_e_turn_signal_state == TurnSignalState.CeSYS_e_LeftTurnSignalOn:
                self.target_relative_lane = RelativeLane.LEFT_LANE
                self.status = LaneChangeStatus.LaneChangeRequested
            elif ego_state.turn_signal.s_Data.e_e_turn_signal_state == TurnSignalState.CeSYS_e_RightTurnSignalOn:
                self.target_relative_lane = RelativeLane.RIGHT_LANE
                self.status = LaneChangeStatus.LaneChangeRequested
        elif self.status == LaneChangeStatus.LaneChangeRequested:
            time_since_lane_change_requested = ego_state.timestamp_in_sec - ego_state.turn_signal.s_Data.s_time_changed.timestamp_in_seconds

            if ego_state.turn_signal.s_Data.e_e_turn_signal_state != LaneChangeState.expected_turn_signal_state[self.target_relative_lane]:
                self._reset()
            elif time_since_lane_change_requested > LANE_CHANGE_DELAY:
                self.status = LaneChangeStatus.AnalyzingSafety
        elif self.status == LaneChangeStatus.AnalyzingSafety:
            if ego_state.turn_signal.s_Data.e_e_turn_signal_state != LaneChangeState.expected_turn_signal_state[self.target_relative_lane]:
                self._reset()
        elif self.status == LaneChangeStatus.LaneChangeActiveInSourceLane:
            # This assumes that if the host has been localized in the target lane, it is definitely over the abort threshold 
            if self._is_lane_id_in_target_lane_ids(ego_state.map_state.lane_id):  # check to see if host has crossed into target lane
                self.status = LaneChangeStatus.LaneChangeActiveInTargetLane
            else:
                dist_to_right_border_in_source_lane, dist_to_left_border_in_source_lane = MapUtils.get_dist_to_lane_borders(
                    ego_state.map_state.lane_id, ego_state.map_state.lane_fstate[FS_SX])

                target_lane_id = MapUtils.get_adjacent_lane_ids(ego_state.map_state.lane_id, self.target_relative_lane)[0]
                dist_to_right_border_in_target_lane, dist_to_left_border_in_target_lane = MapUtils.get_dist_to_lane_borders(
                        target_lane_id, ego_state.map_state.lane_fstate[FS_SX])

                if self.target_relative_lane == RelativeLane.LEFT_LANE:
                    dist_between_lane_centers = dist_to_left_border_in_source_lane + dist_to_right_border_in_target_lane
                    lane_change_percent_complete = max(0.0, ego_state.map_state.lane_fstate[FS_DX] / dist_between_lane_centers * 100.0)
                elif self.target_relative_lane == RelativeLane.RIGHT_LANE:
                    dist_between_lane_centers = dist_to_right_border_in_source_lane + dist_to_left_border_in_target_lane
                    lane_change_percent_complete = max(0.0, -ego_state.map_state.lane_fstate[FS_DX] / dist_between_lane_centers * 100.0)
                else:
                    # We should never get here
                    lane_change_percent_complete = 0.0

                if (ego_state.turn_signal.s_Data.e_e_turn_signal_state
                        != LaneChangeState.expected_turn_signal_state[self.target_relative_lane]
                        and lane_change_percent_complete <= LANE_CHANGE_ABORT_THRESHOLD):
                    self._reset()
        elif self.status == LaneChangeStatus.LaneChangeActiveInTargetLane:
            pass
        elif self.status == LaneChangeStatus.LaneChangeCompleteWaitingForReset:
            if ego_state.turn_signal.s_Data.e_e_turn_signal_state != LaneChangeState.expected_turn_signal_state[self.target_relative_lane]:
                self._reset()

    def update_post_iteration(self, extended_lane_frames: Dict[RelativeLane, GeneralizedFrenetSerretFrame],
                              projected_ego_fstates: Dict[RelativeLane, FrenetState2D], ego_state: EgoState, selected_action: ActionSpec):
        """
        Updates the lane change status after an iteration
        :param extended_lane_frames: dictionary from RelativeLane to the corresponding GeneralizedFrenetSerretFrame
        :param projected_ego_fstates: dictionary from RelativeLane to ego Frenet state, which is ego projected on the corresponding
                                      extended_lane_frame
        :param ego_state: state of host
        :param selected_action: selected action spec
        :return:
        """
        if self.status == LaneChangeStatus.LaneChangeRequestable:
            pass
        elif self.status == LaneChangeStatus.LaneChangeRequested:
            # if lane doesn't exist, reset
            if self.target_relative_lane not in extended_lane_frames.keys():
                self._reset()
        elif self.status == LaneChangeStatus.AnalyzingSafety:
            if self.get_lane_change_mask([selected_action.relative_lane], extended_lane_frames)[0]:
                self.source_lane_gff = extended_lane_frames[RelativeLane.SAME_LANE]
                self._target_lane_ids = extended_lane_frames[selected_action.relative_lane].segment_ids
                self.lane_change_start_time = ego_state.timestamp_in_sec
                self.status = LaneChangeStatus.LaneChangeActiveInSourceLane
        elif self.status == LaneChangeStatus.LaneChangeActiveInSourceLane:
            pass
        elif self.status == LaneChangeStatus.LaneChangeActiveInTargetLane:
            distance_to_target_lane_center = projected_ego_fstates[RelativeLane.SAME_LANE][FS_DX]

            host_station_in_target_lane_gff = np.array([projected_ego_fstates[RelativeLane.SAME_LANE][FS_SX]])
            relative_heading = ego_state.cartesian_state[C_YAW] - \
                               extended_lane_frames[RelativeLane.SAME_LANE].get_yaw(host_station_in_target_lane_gff)[0]

            # If lane change completion requirements are met, the lane change is complete.
            if (abs(distance_to_target_lane_center) < MAX_OFFSET_FOR_LANE_CHANGE_COMPLETE
                    and abs(relative_heading) < MAX_REL_HEADING_FOR_LANE_CHANGE_COMPLETE):
                self.status = LaneChangeStatus.LaneChangeCompleteWaitingForReset
        elif self.status == LaneChangeStatus.LaneChangeCompleteWaitingForReset:
            pass