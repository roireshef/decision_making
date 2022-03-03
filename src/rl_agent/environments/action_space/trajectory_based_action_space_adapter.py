import sys
from abc import ABCMeta, abstractmethod
from typing import List, Dict

import numpy as np
from decision_making.src.global_constants import TRAJECTORY_TIME_RESOLUTION
from decision_making.src.planning.types import FS_SA, FS_SV, FS_DX, FS_SX, FS_2D_LEN, FrenetStates2D
from decision_making.src.planning.utils.optimal_control.poly1d import Poly1D
from decision_making.src.rl_agent.environments.action_space.action_space_adapter import ActionSpaceAdapter
from decision_making.src.rl_agent.environments.action_space.common.data_objects import LaneChangeEvent
from decision_making.src.rl_agent.environments.action_space.common.data_objects import RLActionSpec
from decision_making.src.rl_agent.environments.sim_commands import SimCommand
from decision_making.src.rl_agent.environments.state_space.common.data_objects import EgoCentricState
from decision_making.src.rl_agent.environments.uc_rl_map import RelativeLane
from decision_making.src.rl_agent.utils.frenet_utils import FrenetUtils
from decision_making.src.rl_agent.utils.samplable_werling_trajectory import \
    SamplableWerlingTrajectory


class TrajectoryBasedActionSpaceAdapter(ActionSpaceAdapter, metaclass=ABCMeta):
    """ Abstract class for action space adapters that are based on terminal offset target (laterally) """
    RECIPE_CLS = ActionSpaceAdapter.RECIPE_CLS

    @abstractmethod
    def _get_lateral_targets(self, state: EgoCentricState, action_recipes: List[RECIPE_CLS]) -> \
            (np.ndarray, np.ndarray, List[RelativeLane], List[LaneChangeEvent]):
        """
        Get terminal lateral position dx_T and longitudinal horizon T_d for all action recipes.
        :param state: the current state of environment
        :param action_recipes: the recipes to get the lateral targets for
        :return: Tuple (absolute longitude of terminal goal, time for terminal goal, lane adjacency index relative to
        ego current gff that the lateral offset is <-1, 0, 1>). In cases where lane does not exist or there's an issue
        with creating a lateral motion, the first two are np.nan and the last two are None.
        """
        pass

    @abstractmethod
    def _specify_longitude(self, action_recipes: List[RECIPE_CLS], projected_ego_fstates: FrenetStates2D) -> \
            (np.ndarray, np.ndarray):
        """
        Abstract method to implement that takes action recipes and ego state projection (on the correct target lanes)
        and specifies the longitudinal polynomial and terminal time (horizon) of motion for each recipe
        :param action_recipes: list of N action recipes to specify
        :param projected_ego_fstates: the ego fstates for each recipe projected onto its corresponding target lane frame
        :return: 2D numpy array (shape Nx6) of longitudinal polynomial coefficients, 1D numpy array of terminal times
        """
        pass

    def specify_actions(self, state: EgoCentricState, action_recipes: List[RECIPE_CLS]) -> List[RLActionSpec]:
        """
        This method's purpose is to specify the actions that the agent can take.
        The action_ids are mapped into semantic actions (ActionRecipes) and then translated into terminal state
        specifications (ActionSpecs).
        :param state: a state to use for specifying actions (ego and actors)
        :param action_recipes: a list of action recipes
        :return: semantic action specifications [ActionSpec] or [None] if recipe can't be specified.
        """
        # Get lateral time and offset for terminal state, and reference GFF
        dx_T, T_d, relative_lanes, lane_change_events = self._get_lateral_targets(state, action_recipes)

        # Generate validity mask based on terminal lateral targets specification process
        lat_valid = np.logical_and(~np.isnan(T_d), np.isin(np.array(relative_lanes), state.valid_target_lanes))

        # Project current ego fstate on the target lane for each action
        projected_ego_fstates = np.full((len(action_recipes), FS_2D_LEN), np.nan)
        projected_ego_fstates[lat_valid] = np.array([
            state.ego_fstate_on_adj_lanes[relative_lane]
            for relative_lane, is_valid in zip(relative_lanes, lat_valid) if is_valid
        ])

        # Generate lateral polynomial coefficients (solved in target lane's reference frame)
        sx_0, sv_0, sa_0, dx_0, dv_0, da_0 = projected_ego_fstates.transpose()
        poly_coefs_d = FrenetUtils.generate_polynomials_1d(dx_0, dv_0, da_0, dx_T, np.zeros_like(dv_0), T_d)

        # Get longitudinal time and offset for terminal state. Note that NaNs are result of root not find in range
        valid_recipes = [recipe for recipe, valid in zip(action_recipes, lat_valid) if valid]
        poly_coefs_s, T_s = np.full((len(action_recipes), 6), np.nan), np.full(len(action_recipes), np.nan)
        poly_coefs_s[lat_valid], T_s[lat_valid] = self._specify_longitude(valid_recipes, projected_ego_fstates[lat_valid])
        sx_T, sv_T, _ = Poly1D.zip_polyval_with_derivatives(poly_coefs_s, T_s[:, np.newaxis]).squeeze(1).transpose()

        is_valid_on_both_dims = ~np.isnan(T_s)  # this includes lateral and longitudinal inside

        try:
            gff_adj_dict = state.roads_map.gff_adjacency[state.ego_state.gff_id]
            action_specs = [
                RLActionSpec(max(ts, td), vt, st, dt, relative_lane, recipe,
                             baseline_trajectory=SamplableWerlingTrajectory(
                                 timestamp_in_sec=state.timestamp_in_sec, T_s=ts, T_d=td,
                                 T_extended=self._params["ACTION_MAX_TIME_HORIZON"],
                                 poly_s_coefs=poly_s, poly_d_coefs=poly_d,
                                 gff_id=gff_adj_dict[relative_lane]),
                             lane_change_event=lc_event)
                if is_valid else None
                for recipe, vt, st, ts, dt, td, relative_lane, poly_s, poly_d, lc_event, is_valid
                in zip(action_recipes, sv_T, sx_T, T_s, dx_T, T_d, relative_lanes, poly_coefs_s, poly_coefs_d,
                       lane_change_events, is_valid_on_both_dims)
            ]
        except Exception as e:
            raise type(e)(str(e) + '; local variables: %s ' % locals()).with_traceback(sys.exc_info()[2])

        return action_specs

    def get_commands(self, state: EgoCentricState, action_id: int) -> (List[SimCommand], Dict):
        """
        Given a State object and and Action ID, this method "specifies" the action, builds a trajectory for it, and
        finally samples points along the trajectory to get acceleration samples that serve as commands for the
        simulation
        :param state: a state (to use for initial ego vehicle state)
        :param action_id: The id of the action in this action space
        :return: tuple(list of acceleration commands, lateral command, debug information)
        """
        action_recipe = self.action_recipes[action_id]

        # specify the action
        action_spec = self.specify_action(state, action_recipe)

        # build a baseline trajectory for the action
        samplable_trajectory = action_spec.baseline_trajectory

        # sample accelerations and velocities from the trajectory
        relative_time_points = np.linspace(0, self._params["SIMS_PER_STEP"] * TRAJECTORY_TIME_RESOLUTION,
                                           self._params["SIMS_PER_STEP"] + 1)
        trajectory = samplable_trajectory.sample_frenet(time_points=relative_time_points+state.timestamp_in_sec)
        traj_vels, traj_accs, traj_lat_pos, traj_lon_pos = \
            trajectory[:self._params["SIMS_PER_STEP"] + 1, [FS_SV, FS_SA, FS_DX, FS_SX]].T

        commands = [SimCommand(vel, 0) for vel in traj_vels[1:]]

        projected_ego_fstate = state.ego_fstate_on_adj_lanes[action_spec.relative_lane]
        commands[0] = SimCommand(commands[0].target_speed, traj_lat_pos[-1] - projected_ego_fstate[FS_DX])

        intermediate_jerk = np.polyval(np.polyder(samplable_trajectory.poly_s_coefs, 3), relative_time_points)

        info = {"traj_vels": traj_vels[1:],
                "traj_accs": traj_accs[1:],
                "traj_stations": traj_lon_pos[1:],
                "action_spec": action_spec,
                "action_not_none_nor_short": action_spec is not None and action_spec.t > TRAJECTORY_TIME_RESOLUTION,
                "action_spec_none": action_spec is None,
                "intermediate_jerk": intermediate_jerk[1:],
                "knot_jerk_diff": intermediate_jerk[0] - (state.ego_jerk or 0),
                "sum_sq_lon_acc": samplable_trajectory.get_lon_sq_acc_at(np.array([0, 1])),
                "sum_sq_lon_jerk": samplable_trajectory.get_lon_sq_jerk_at(np.array([0, 1])),
                "sum_sq_lat_jerk": samplable_trajectory.get_lat_sq_jerk_at(np.array([0, 1]))}

        return commands, info
