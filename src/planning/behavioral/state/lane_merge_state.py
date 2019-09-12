from decision_making.src.planning.behavioral.state.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import RelativeLane
import numpy as np
from decision_making.src.planning.types import FP_SX

MAX_BACK_HORIZON = 300   # on the main road
MAX_AHEAD_HORIZON = 100  # on the main road
MERGE_LOOKAHEAD = 300    # on the ego road


class LaneMergeState(BehavioralGridState):
    def __init__(self, behavioral_state: BehavioralGridState,
                 merge_side: RelativeLane, can_merge_from_s: float, red_line_s):
        super().__init__(behavioral_state.road_occupancy_grid, behavioral_state.ego_state,
                         behavioral_state.extended_lane_frames, behavioral_state.projected_ego_fstates)
        self.merge_side = merge_side
        self.can_merge_from_s = can_merge_from_s
        self.red_line_s = red_line_s

    @classmethod
    def create_from_behavioral_state(cls, behavioral_state: BehavioralGridState):
        ego_state = behavioral_state.ego_state
        ego_gff = behavioral_state.extended_lane_frames[RelativeLane.SAME_LANE]
        same_lane_segments = ego_gff.segment_ids
        same_lane_offsets = ego_gff._segments_s_offsets

        for rel_lane in [RelativeLane.RIGHT_LANE, RelativeLane.LEFT_LANE]:
            target_gff = behavioral_state.extended_lane_frames[rel_lane]
            if target_gff.gff_type != GFF_Type.Augmented:
                continue
            current_index = np.where(same_lane_segments == ego_state.map_state.lane_id)[0][0]
            max_index = min(len(same_lane_segments), len(target_gff.segment_ids))
            merge_indices = np.where(target_gff.segment_ids[:max_index] == same_lane_segments[:max_index])[0]
            if len(merge_indices) == 0 or merge_indices[0] <= current_index:  # the merge point should be ahead
                continue
            merge_side = rel_lane
            can_merge_from_s = same_lane_offsets[merge_indices[0]]
            red_line_s = same_lane_offsets[merge_indices[0] - 1] - ego_state.size.length / 2
            return cls(behavioral_state, merge_side, can_merge_from_s, red_line_s)

        return None

    # @staticmethod
    # def build(state: State, behavioral_state: BehavioralGridState):
    #     """
    #     Given the current state, find the lane merge ahead and cars on the main road upstream the merge point.
    #     :param state: current state
    #     :param behavioral_state: current behavioral grid state
    #     :return: lane merge state or None (if no merge), s of merge-point in GFF
    #     """
    #     gff = behavioral_state.extended_lane_frames[RelativeLane.SAME_LANE]
    #     ego_fstate_on_gff = behavioral_state.projected_ego_fstates[RelativeLane.SAME_LANE]
    #
    #     # Find the lanes before and after the merge point
    #     merge_lane_id = MapUtils.get_closest_lane_merge(gff, ego_fstate_on_gff[FS_SX], merge_lookahead=MERGE_LOOKAHEAD)
    #     if merge_lane_id is None:
    #         return None, 0
    #     after_merge_lane_id = MapUtils.get_downstream_lanes(merge_lane_id)[0]
    #     main_lane_ids, main_lanes_s = MapUtils.get_straight_upstream_downstream_lanes(
    #         after_merge_lane_id, max_back_horizon=MAX_BACK_HORIZON, max_ahead_horizon=MAX_AHEAD_HORIZON)
    #     if len(main_lane_ids) == 0:
    #         return None, 0
    #
    #     # calculate s of the red line, as s on GFF of the merge lane segment origin
    #     red_line_in_gff = gff.convert_from_segment_state(np.zeros(FS_2D_LEN), merge_lane_id)[FS_SX]
    #     # calculate s of the merge point, as s on GFF of segment origin of after-merge lane
    #     merge_point_in_gff = gff.convert_from_segment_state(np.zeros(FS_2D_LEN), after_merge_lane_id)[FS_SX]
    #     # calculate distance from ego to the merge point
    #     dist_to_merge_point = merge_point_in_gff - ego_fstate_on_gff[FS_SX]
    #
    #     # check existence of cars on the upstream main road
    #     actors = []
    #     main_lane_ids_arr = np.array(main_lane_ids)
    #     for obj in state.dynamic_objects:
    #         if obj.map_state.lane_id in main_lane_ids:
    #             lane_idx = np.where(main_lane_ids_arr == obj.map_state.lane_id)[0][0]
    #             obj_s = main_lanes_s[lane_idx] + obj.map_state.lane_fstate[FS_SX]
    #             if -MAX_BACK_HORIZON < obj_s < MAX_AHEAD_HORIZON:
    #                 obj_fstate = np.concatenate(([obj_s], obj.map_state.lane_fstate[FS_SV:]))
    #                 actors.append(ActorState(obj.size, obj_fstate))
    #
    #     ego_in_lane_merge = np.array([-dist_to_merge_point, ego_fstate_on_gff[FS_SV], ego_fstate_on_gff[FS_SA]])
    #     return LaneMergeState(ego_in_lane_merge, behavioral_state.ego_state.size, actors,
    #                           red_line_in_gff - merge_point_in_gff, merge_point_in_gff)

