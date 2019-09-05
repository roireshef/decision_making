from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import RelativeLane
from decision_making.src.planning.types import FrenetState1D, FS_SX, FS_2D_LEN, FS_SV, FS_SA
from decision_making.src.state.state import ObjectSize, State
from decision_making.src.utils.map_utils import MapUtils
from typing import List
import numpy as np


MAX_BACK_HORIZON = 300   # on the main road
MAX_AHEAD_HORIZON = 100  # on the main road
MERGE_LOOKAHEAD = 300    # on the ego road


class ActorState:
    def __init__(self, size: ObjectSize, fstate: FrenetState1D):
        self.size = size
        self.fstate = fstate


class LaneMergeState:
    def __init__(self, ego_fstate: FrenetState1D, ego_size: ObjectSize, main_road_actors: List[ActorState],
                 merge_point_red_line_dist: float, merge_point_s_in_gff: float):
        self.ego_fstate = ego_fstate  # SX is negative: -dist_to_red_line
        self.ego_size = ego_size
        self.main_road_actors = main_road_actors
        self.merge_point_red_line_dist = merge_point_red_line_dist
        self.merge_point_in_gff = merge_point_s_in_gff

    @staticmethod
    def build(state: State, behavioral_state: BehavioralGridState):
        """
        Given the current state, find the lane merge ahead and cars on the main road upstream the merge point.
        :param state: current state
        :param behavioral_state: current behavioral grid state
        :return: lane merge state or None (if no merge), s of merge-point in GFF
        """
        gff = behavioral_state.extended_lane_frames[RelativeLane.SAME_LANE]
        ego_fstate_on_gff = behavioral_state.projected_ego_fstates[RelativeLane.SAME_LANE]

        # Find the lanes before and after the merge point
        merge_lane_id = MapUtils.get_closest_lane_merge(gff, ego_fstate_on_gff[FS_SX], merge_lookahead=MERGE_LOOKAHEAD)
        if merge_lane_id is None:
            return None, 0
        after_merge_lane_id = MapUtils.get_downstream_lanes(merge_lane_id)[0]
        main_lane_ids, main_lanes_s = MapUtils.get_straight_upstream_downstream_lanes(
            after_merge_lane_id, max_back_horizon=MAX_BACK_HORIZON, max_ahead_horizon=MAX_AHEAD_HORIZON)
        if len(main_lane_ids) == 0:
            return None, 0

        # calculate s of the red line, as s on GFF of the merge lane segment origin
        red_line_in_gff = gff.convert_from_segment_state(np.zeros(FS_2D_LEN), merge_lane_id)[FS_SX]
        # calculate s of the merge point, as s on GFF of segment origin of after-merge lane
        merge_point_in_gff = gff.convert_from_segment_state(np.zeros(FS_2D_LEN), after_merge_lane_id)[FS_SX]
        # calculate distance from ego to the merge point
        dist_to_merge_point = merge_point_in_gff - ego_fstate_on_gff[FS_SX]

        # check existence of cars on the upstream main road
        actors = []
        main_lane_ids_arr = np.array(main_lane_ids)
        for obj in state.dynamic_objects:
            if obj.map_state.lane_id in main_lane_ids:
                lane_idx = np.where(main_lane_ids_arr == obj.map_state.lane_id)[0][0]
                obj_s = main_lanes_s[lane_idx] + obj.map_state.lane_fstate[FS_SX]
                if -MAX_BACK_HORIZON < obj_s < MAX_AHEAD_HORIZON:
                    obj_fstate = np.concatenate(([obj_s], obj.map_state.lane_fstate[FS_SV:]))
                    actors.append(ActorState(obj.size, obj_fstate))

        ego_in_lane_merge = np.array([-dist_to_merge_point, ego_fstate_on_gff[FS_SV], ego_fstate_on_gff[FS_SA]])
        return LaneMergeState(ego_in_lane_merge, behavioral_state.ego_state.size, actors,
                              merge_point_in_gff - red_line_in_gff, merge_point_in_gff)

