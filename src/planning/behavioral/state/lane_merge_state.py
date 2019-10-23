from logging import Logger

from decision_making.src.exceptions import MappingException
from decision_making.src.global_constants import MERGE_LOOKAHEAD
from decision_making.src.messages.route_plan_message import RoutePlan
from decision_making.src.messages.scene_static_enums import ManeuverType
from decision_making.src.planning.behavioral.state.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import RelativeLane, RelativeLongitudinalPosition
import numpy as np
from decision_making.src.planning.types import FS_DX, FS_SX, FS_2D_LEN, FS_SV, FS_1D_LEN
from decision_making.src.state.state import State
from decision_making.src.utils.map_utils import MapUtils
from gym.spaces.tuple_space import Tuple as GymTuple
from planning_research.src.flow_rl.common_constants import DEFAULT_ADDITIONAL_ENV_PARAMS  # TODO: move from planning research
import torch


class LaneMergeState(BehavioralGridState):
    def __init__(self, road_occupancy_grid, ego_state, extended_lane_frames, projected_ego_fstates,
                 target_relative_lane: RelativeLane, red_line_s: float):
        super().__init__(road_occupancy_grid, ego_state, extended_lane_frames, projected_ego_fstates)
        self.target_relative_lane = target_relative_lane
        self.red_line_s = red_line_s

    @classmethod
    def create_from_state(cls, state: State, route_plan: RoutePlan, logger: Logger):
        """
        Create LaneMergeState from a given State.
        The output state has two GFFs: for same lane and for the target lane.
        Actors's longitudinal distances from ego are aligned to the merge point that is common to both GFFs.
        :param state: current state
        :param route_plan: route plan
        :param logger:
        :return: LaneMergeState
        """
        ego_state = state.ego_state
        ego_lane_id, ego_lane_fstate = ego_state.map_state.lane_id, ego_state.map_state.lane_fstate

        # find merge lane_id of ego_gff, merge side and the first common lane_id
        merge_lane_id, maneuver_type, common_lane_id = MapUtils.get_closest_lane_merge(
            ego_lane_id, ego_lane_fstate[FS_SX], MERGE_LOOKAHEAD, route_plan)

        target_relative_lane = RelativeLane.RIGHT_LANE if maneuver_type == ManeuverType.LEFT_MERGE_CONNECTION else RelativeLane.LEFT_LANE

        try:
            # create GFF for the host's lane
            ego_gff = BehavioralGridState._get_generalized_frenet_frames(
                lane_id=ego_lane_id, station=ego_lane_fstate[FS_SX], route_plan=route_plan)[RelativeLane.SAME_LANE]

            # project ego on its GFF
            ego_on_same_gff = ego_gff.convert_from_segment_state(ego_lane_fstate, ego_lane_id)

            # set red line s to be the origin of merge_lane_id (the last lane segment before the merge point)
            red_line_s = ego_gff.convert_from_segment_state(np.zeros(FS_2D_LEN), merge_lane_id)[FS_SX]

            # calculate merge point s relative to ego
            merge_point_on_ego_gff = ego_gff.convert_from_segment_state(np.zeros(FS_2D_LEN), common_lane_id)[FS_SX]
            merge_point_from_ego = merge_point_on_ego_gff - ego_on_same_gff[FS_SX]

            # create target GFF for the merge, such that its backward & forward horizons are equal to MERGE_LOOKAHEAD
            # relative to ego
            target_gff = BehavioralGridState._get_generalized_frenet_frames(
                common_lane_id, station=0, route_plan=route_plan, forward_horizon=MERGE_LOOKAHEAD - merge_point_from_ego,
                backward_horizon=MERGE_LOOKAHEAD + merge_point_from_ego)[RelativeLane.SAME_LANE]

            all_gffs = {RelativeLane.SAME_LANE: ego_gff, target_relative_lane: target_gff}

            # Project ego on target_GFF, such that the projection has the same distance to the merge point as ego
            # itself. The lateral parameters of the projection are zeros.
            ego_on_target_gff = np.concatenate((ego_on_same_gff[:FS_DX], np.zeros(FS_1D_LEN)))
            merge_point_on_target_gff = target_gff.convert_from_segment_state(np.zeros(FS_2D_LEN), common_lane_id)[FS_SX]
            ego_on_target_gff[FS_SX] += merge_point_on_target_gff - merge_point_on_ego_gff
            projected_ego = {RelativeLane.SAME_LANE: ego_on_same_gff, target_relative_lane: ego_on_target_gff}

            # create road_occupancy_grid by using the appropriate BehavioralGridState methods
            actors_with_road_semantics = \
                sorted(BehavioralGridState._add_road_semantics(state.dynamic_objects, all_gffs, projected_ego),
                       key=lambda rel_obj: abs(rel_obj.longitudinal_distance))
            multi_object_grid = BehavioralGridState._project_objects_on_grid(actors_with_road_semantics, ego_state)

            return cls(road_occupancy_grid=multi_object_grid, ego_state=ego_state,
                       extended_lane_frames=all_gffs, projected_ego_fstates=projected_ego,
                       target_relative_lane=target_relative_lane, red_line_s=red_line_s)

        except MappingException as e:
            # in case of failure to build GFF for SAME_LANE or target lane GFF, stop processing this BP frame
            raise AssertionError("Trying to fetch data for %s, but data is unavailable. %s" % (RelativeLane.SAME_LANE, str(e)))

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
                            for obj in self.road_occupancy_grid[(self.target_relative_lane, lon_cell)]
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
