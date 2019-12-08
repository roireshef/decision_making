from typing import List

import numpy as np
from decision_making.src.global_constants import LANE_MERGE_STATE_FAR_AWAY_DISTANCE, \
    LANE_MERGE_STATE_OCCUPANCY_GRID_ONESIDED_LENGTH, LANE_MERGE_STATE_OCCUPANCY_GRID_RESOLUTION, \
    LANE_MERGE_ACTION_SPACE_MAX_VELOCITY
from decision_making.src.planning.behavioral.state.lane_merge_actor_state import LaneMergeActorState
from decision_making.src.planning.types import FS_SX, FrenetState1D, FS_SV, \
    FS_SA
from gym.spaces import Tuple as GymTuple


class LaneMergeStateEncoder:
    @staticmethod
    def encode_state(ego_fstate_1d: FrenetState1D, red_line_s_on_ego_gff: float,
                     actors_states: List[LaneMergeActorState]) -> GymTuple:
        """
        Encode and normalize the LaneMergeState for RL model usage.
        :param ego_fstate_1d: 1-dim longitudinal Frenet state of ego
        :param red_line_s_on_ego_gff: s of the red line on host's GFF
        :param actors_states: list of actor states
        :return: tuple of host state and actors state (of type torch.tensor)
        """
        # normalize host & actors states
        host_state = LaneMergeStateEncoder._encode_host_state(ego_fstate_1d, red_line_s_on_ego_gff) / \
                     np.array([LANE_MERGE_STATE_FAR_AWAY_DISTANCE, LANE_MERGE_ACTION_SPACE_MAX_VELOCITY, 1])
        actors_state = LaneMergeStateEncoder._encode_actors_state(actors_states) / \
                       np.array([1, LANE_MERGE_ACTION_SPACE_MAX_VELOCITY])[..., np.newaxis]
        return host_state, actors_state


    @staticmethod
    def _encode_host_state(ego_fstate_1d: FrenetState1D, red_line_s_on_ego_gff: float) -> np.array:
        """
        Encode host state of the LaneMergeState for RL model usage.
        :param ego_fstate_1d: 1-dim longitudinal Frenet state of ego
        :param red_line_s_on_ego_gff: s of the red line on host's GFF
        :return: numpy array with encoded host state
        """
        # encode host: replace the host station coordinate with its distance to red line
        host_state = np.array([red_line_s_on_ego_gff - ego_fstate_1d[FS_SX], ego_fstate_1d[FS_SV], ego_fstate_1d[FS_SA]])
        return host_state[np.newaxis, :]


    @staticmethod
    def _encode_actors_state(actors_states: List[LaneMergeActorState]) -> np.array:
        """
        Encode actors state of the LaneMergeState for RL model usage.
        In the current implementation the road is divided into a grid with resolution
        LANE_MERGE_STATE_OCCUPANCY_GRID_RESOLUTION, and each cell is set 1 iff an actor appears in the cell.
        In addition, the actor's velocity is appended to the appropriated cell.
        :param actors_states: list of actor states
        :return: numpy array with encoded actors state
        """
        # actors state is an occupancy grid containing the different vehicles' distance from merge and velocity
        num_of_onesided_grid_cells = np.ceil(LANE_MERGE_STATE_OCCUPANCY_GRID_ONESIDED_LENGTH /
                                             LANE_MERGE_STATE_OCCUPANCY_GRID_RESOLUTION).astype(int)
        num_of_grid_cells = 2 * num_of_onesided_grid_cells

        # init for empty grid cells
        actors_exist_default = np.zeros(shape=(1, num_of_grid_cells))
        actors_vel_default = -LANE_MERGE_ACTION_SPACE_MAX_VELOCITY * np.ones(shape=(1, num_of_grid_cells))
        encoded_actors_states = np.vstack((actors_exist_default, actors_vel_default))

        for actor in actors_states:
            actor_exists = 1
            actor_grid_cell = np.floor(actor.s_relative_to_ego / LANE_MERGE_STATE_OCCUPANCY_GRID_RESOLUTION). \
                                  astype(int) + num_of_onesided_grid_cells
            if 0 <= actor_grid_cell <= num_of_grid_cells - 1:
                encoded_actors_states[:, actor_grid_cell] = np.array([actor_exists, actor.velocity])

        return encoded_actors_states
