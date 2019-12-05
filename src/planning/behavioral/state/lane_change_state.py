from enum import Enum
import numpy as np
from typing import Optional, Dict

from decision_making.src.global_constants import MAX_OFFSET_FOR_LANE_CHANGE_COMPLETE, MAX_REL_HEADING_FOR_LANE_CHANGE_COMPLETE
from decision_making.src.messages.turn_signal_message import TurnSignalState
from decision_making.src.planning.behavioral.data_objects import ActionSpec, RelativeLane
from decision_making.src.planning.types import FS_DX, FS_SX, C_YAW, FrenetState2D
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GFFType, GeneralizedFrenetSerretFrame
from decision_making.src.state.state import EgoState



class LaneChangeStatus(Enum):
    LaneChangeRequestable = 0   # TODO: Find a better name so that it's not confused with LaneChangeRequested?
    LaneChangeRequested = 1
    AnalyzingSafety = 2
    LaneChangeActiveInSourceLane = 3
    LaneChangeActiveInTargetLane = 4
    LaneChangeCompleteWaitingForReset = 5


class LaneChangeState:
    def __init__(self, source_lane_gff: Optional[GeneralizedFrenetSerretFrame] = None, target_lane_ids: Optional[np.ndarray] = None,
                 lane_change_active: Optional[bool] = False, lane_change_start_time: Optional[float] = None,
                 status: Optional[LaneChangeStatus] = LaneChangeStatus.LaneChangeRequestable):
        """
        Holds lane change state
        :param source_lane_gff: GFF that the host was in when a lane change was initiated
        :param target_lane_ids: Lane IDs of the GFF that the host is targeting in a lane change
        :param lane_change_active: True when a lane change is active; otherwise, False
        :param lane_change_start_time: Time when a lane change began
        :param status: lane change status
        """
        self.source_lane_gff = source_lane_gff
        self._target_lane_ids = target_lane_ids or np.array([])
        self.lane_change_active = lane_change_active
        self.lane_change_start_time = lane_change_start_time
        self.status = status

    def __str__(self):
        # print as dict for logs
        return str(self.__dict__)

    def _reset(self):
        self.source_lane_gff = None
        self._target_lane_ids = np.array([])
        self.lane_change_active = False
        self.lane_change_start_time = None

    def are_target_lane_ids_in_gff(self, gff: GeneralizedFrenetSerretFrame):
        """
        This checks whether the lane IDs in the target GFF are in the given GFF. This can be used to see whether the host has crossed into
        the target lane during a lane change.
        :param gff:
        :return: Returns True if any lane IDs in the target GFF are in the given GFF
        """
        return np.any(np.isin(self._target_lane_ids, gff.segment_ids))

    def update(self, extended_lane_frames: Dict[RelativeLane, GeneralizedFrenetSerretFrame],
               projected_ego_fstates: Dict[RelativeLane, FrenetState2D], ego_state: EgoState, selected_action: ActionSpec):
        """
        Update the lane change status
        :param extended_lane_frames: dictionary from RelativeLane to the corresponding GeneralizedFrenetSerretFrame
        :param projected_ego_fstates: dictionary from RelativeLane to ego Frenet state, which is ego projected on the corresponding
                                      extended_lane_frame
        :param ego_state: state of host
        :param selected_action: selected action spec
        :return:
        """
        if self.lane_change_active:
            # If lane change is currently active, check to see whether lane change is complete
            is_host_in_target_lane = self.are_target_lane_ids_in_gff(extended_lane_frames[RelativeLane.SAME_LANE])

            if is_host_in_target_lane:
                distance_to_target_lane_center = projected_ego_fstates[RelativeLane.SAME_LANE][FS_DX]

                host_station_in_target_lane_gff = np.array([projected_ego_fstates[RelativeLane.SAME_LANE][FS_SX]])
                relative_heading = ego_state.cartesian_state[C_YAW] - \
                                   extended_lane_frames[RelativeLane.SAME_LANE].get_yaw(host_station_in_target_lane_gff)[0]

                # If lane change completion requirements are met and the turn signal has been turned off, the lane change is complete.
                if (abs(distance_to_target_lane_center) < MAX_OFFSET_FOR_LANE_CHANGE_COMPLETE
                        and abs(relative_heading) < MAX_REL_HEADING_FOR_LANE_CHANGE_COMPLETE
                        and ego_state.turn_signal.s_Data.e_e_turn_signal_state == TurnSignalState.CeSYS_e_Off):
                    self._reset()
        else:
            # If lane change is not currently active, check to see whether a lane change action was selected
            if (selected_action.relative_lane in [RelativeLane.LEFT_LANE, RelativeLane.RIGHT_LANE]
                and extended_lane_frames[selected_action.relative_lane].gff_type not in
                    [GFFType.Augmented, GFFType.AugmentedPartial]):
                self.lane_change_active = True
                self.source_lane_gff = extended_lane_frames[RelativeLane.SAME_LANE]
                self._target_lane_ids = extended_lane_frames[selected_action.relative_lane].segment_ids
                self.lane_change_start_time = ego_state.timestamp_in_sec
