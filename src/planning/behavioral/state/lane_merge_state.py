from decision_making.src.global_constants import MERGE_LOOKAHEAD
from decision_making.src.messages.route_plan_message import RoutePlan
from decision_making.src.messages.scene_static_enums import ManeuverType
from decision_making.src.planning.behavioral.state.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import RelativeLane, RelativeLongitudinalPosition
import numpy as np
from decision_making.src.planning.types import FS_DX, FS_SX, FS_2D_LEN
from decision_making.src.utils.map_utils import MapUtils
from gym.spaces.tuple_space import Tuple as GymTuple
from planning_research.src.flow_rl.common_constants import DEFAULT_ADDITIONAL_ENV_PARAMS  # TODO: move from planning research
import torch


class LaneMergeState(BehavioralGridState):
    def __init__(self, behavioral_state: BehavioralGridState,
                 merge_side: RelativeLane, red_line_s: float, lanes_s_diff: float):
        super().__init__(behavioral_state.road_occupancy_grid, behavioral_state.ego_state,
                         behavioral_state.extended_lane_frames, behavioral_state.projected_ego_fstates)
        self.merge_side = merge_side
        self.red_line_s = red_line_s
        self.lanes_s_diff = lanes_s_diff

    @classmethod
    def create_from_behavioral_state(cls, behavioral_state: BehavioralGridState, route_plan: RoutePlan):
        ego_gff = behavioral_state.extended_lane_frames[RelativeLane.SAME_LANE]
        projected_ego = behavioral_state.projected_ego_fstates[RelativeLane.SAME_LANE]

        # find merge lane_id of ego_gff, merge side and the first common lane_id
        merge_lane_id, maneuver_type, common_lane_id = MapUtils.get_closest_lane_merge(
            ego_gff, projected_ego[FS_SX], merge_lookahead=MERGE_LOOKAHEAD)
        relative_lane = RelativeLane.LEFT_LANE if maneuver_type == ManeuverType.LEFT_MERGE_CONNECTION else RelativeLane.RIGHT_LANE

        red_line_s = ego_gff.convert_from_segment_state(np.zeros(FS_2D_LEN), merge_lane_id)[FS_SX]
        merge_point_on_ego_gff = ego_gff.convert_from_segment_state(np.zeros(FS_2D_LEN), common_lane_id)[FS_SX]
        merge_point_from_ego = merge_point_on_ego_gff - projected_ego[FS_SX]

        # create target GFF for the merge
        target_gff = behavioral_state._get_generalized_frenet_frames(
            common_lane_id, station=0, route_plan=route_plan, forward_horizon=MERGE_LOOKAHEAD - merge_point_from_ego,
            backward_horizon=MERGE_LOOKAHEAD + merge_point_from_ego)[RelativeLane.SAME_LANE]

        merge_point_on_target_gff = target_gff.convert_from_segment_state(np.zeros(FS_2D_LEN), common_lane_id)[FS_SX]
        lanes_s_diff = merge_point_on_target_gff - merge_point_on_ego_gff

        return cls(behavioral_state, relative_lane, red_line_s, lanes_s_diff)

    def encode_state_for_RL(self) -> GymTuple:

        # encode host
        host_state = self.projected_ego_fstates[RelativeLane.SAME_LANE][:FS_DX]
        # replace the host station coordinate with its distance to merging point (host_dist_from_merge can be negative,
        # meaning host is past the red line (usually the episode is done before that)
        host_state[FS_SX] = self.red_line_s - host_state[FS_SX]

        params = DEFAULT_ADDITIONAL_ENV_PARAMS
        grid_res = params["OCCUPANCY_GRID_RESOLUTION"]
        grid_onesided_length = params["OCCUPANCY_GRID_ONESIDED_LENGTH"]

        # encode actors
        perceived_actors = [obj for lon_cell in RelativeLongitudinalPosition
                            for obj in self.road_occupancy_grid[(self.merge_side, lon_cell)]
                            if obj is not None and abs(obj.longitudinal_distance) < grid_onesided_length]

        # actors state is an occupancy grid containing the different vehicles' distance from merge and velocity
        num_of_onesided_grid_cells = np.ceil(grid_onesided_length / grid_res).astype(int)
        num_of_grid_cells = 2 * num_of_onesided_grid_cells

        # init for empty grid cells
        actors_exist_default = np.zeros(shape=(num_of_grid_cells, 1))
        actors_vel_default = -params["MAX_VELOCITY"] * np.ones(shape=(num_of_grid_cells, 1))
        actors_state = np.hstack((actors_exist_default, actors_vel_default))

        for actor in perceived_actors:
            actor_exists = 1
            actor_grid_cell = np.floor(actor.longitudinal_distance / grid_res).astype(int) + num_of_onesided_grid_cells
            if 0 <= actor_grid_cell <= num_of_grid_cells - 1:
                actors_state[actor_grid_cell] = np.array([actor_exists, actor.dynamic_object.velocity])

        return torch.from_numpy(host_state[np.newaxis, np.newaxis, :]).float(), torch.from_numpy(actors_state[np.newaxis, :]).float()

    # MAX_BACK_HORIZON = 300   # on the main road
    # MAX_AHEAD_HORIZON = 100  # on the main road
    #
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

