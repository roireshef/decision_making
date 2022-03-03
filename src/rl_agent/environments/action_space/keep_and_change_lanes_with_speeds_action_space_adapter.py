from itertools import product
from typing import List, Dict

import numpy as np
from decision_making.src.planning.types import FS_DX, FS_SX, FS_DV, FS_DA
from decision_making.src.planning.utils.numpy_utils import UniformGrid
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D
from decision_making.src.rl_agent.environments.action_space.common.data_objects import \
    LateralOffsetTerminalVelocityActionRecipe
from decision_making.src.rl_agent.environments.action_space.common.lateral_mixins import CommitLaneChangeMixin
from decision_making.src.rl_agent.environments.action_space.common.longitudinal_mixins import TerminalVelocityMixin
from decision_making.src.rl_agent.environments.action_space.trajectory_based_action_space_adapter import \
    TrajectoryBasedActionSpaceAdapter
from decision_making.src.rl_agent.environments.sim_commands import SimCommand
from decision_making.src.rl_agent.environments.state_space.common.data_objects import EgoCentricState
from decision_making.src.rl_agent.environments.uc_rl_map import MapAnchor
from decision_making.src.rl_agent.environments.uc_rl_map import RelativeLane
from decision_making.src.rl_agent.global_types import LateralDirection


class KeepAndChangeLanesWithSpeedActionSpaceAdapter(CommitLaneChangeMixin, TerminalVelocityMixin,
                                                    TrajectoryBasedActionSpaceAdapter):
    RECIPE_CLS = LateralOffsetTerminalVelocityActionRecipe

    def __init__(self, action_space_params) -> None:
        """ Implements Action Space (with methods) """

        velocity_grid = UniformGrid(limits=np.array([action_space_params['MIN_VELOCITY'],
                                                     action_space_params['MAX_VELOCITY']]),
                                    resolution=action_space_params['VELOCITY_RESOLUTION'])
        super().__init__(action_space_params,
                         [LateralOffsetTerminalVelocityActionRecipe(*comb)
                          for comb in product(list(LateralDirection.__iter__()),
                                              velocity_grid.array,
                                              action_space_params['AGGRESSIVENESS_LEVELS'])])


class KeepAndChangeLanesWithSpeedActionSpaceAdapterForLaneMerge(KeepAndChangeLanesWithSpeedActionSpaceAdapter):
    """ This action-space generate all actions of FullTrajectoryNegotiationActionSpaceAdapter, but when crossing into
    an area of two lanes merging together, it adds a penalty that acts as a proxy to the cartesian Jerk applied when
    driving through a curve """

    def get_commands(self, state: EgoCentricState, action_id: int) -> (List[SimCommand], Dict):
        commands, info = super().get_commands(state, action_id)

        # get end the station at the horizon (decision making rate, e.g. 1 sec ahead)
        sx_T = info["traj_stations"][-1]

        # this parameters acts as the proxy for the duration of the "forced" lane change applied during the merge
        forced_lc_duration = self._params["FORCED_LC_DURATION"]

        # TODO: sx_T is represented in terms of target lane and map anchor in terms of current lane
        if forced_lc_duration > 0 and state.ego_state.gff_id == 'ramp_gff' and \
                state.ego_state.fstate[FS_SX] <= state.self_gff_anchors[MapAnchor.YIELD_LINE] < sx_T:
            # get lateral coordinates on the target GFF (left-lane)
            fstate_on_target = state.ego_fstate_on_adj_lanes[RelativeLane.LEFT_LANE]
            dx_0, dv_0, da_0 = fstate_on_target[[FS_DX, FS_DV, FS_DA]]

            # solve 2p-bvp for converging to target lane's center
            A_inv = QuinticPoly1D.inverse_time_constraints_matrix(forced_lc_duration)
            constraints = np.array([dx_0, dv_0, da_0, 0, 0, 0])
            forced_lc_poly_d = QuinticPoly1D.solve(A_inv, constraints[np.newaxis, :])[0]

            # add "forced" jerk to the info dict
            forced_lat_jerk = QuinticPoly1D.sq_jerk_between(forced_lc_poly_d, 0, forced_lc_duration).item()
            info["sum_sq_lat_jerk"] += forced_lat_jerk

        return commands, info
