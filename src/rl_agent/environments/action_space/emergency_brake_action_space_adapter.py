from abc import ABCMeta
from typing import List, Dict

import numpy as np
from decision_making.src.global_constants import TRAJECTORY_TIME_RESOLUTION
from decision_making.src.planning.types import FS_SA, FS_SV, FS_DX, FrenetStates2D
from decision_making.src.rl_agent.environments.action_space.common.data_objects import RLActionRecipe, \
    ConstantAccelerationRLActionRecipe
from decision_making.src.rl_agent.environments.action_space.common.lateral_mixins import CommitLaneChangeMixin, \
    NegotiateLaneChangeMixin
from decision_making.src.rl_agent.environments.action_space.trajectory_based_action_space_adapter import \
    TrajectoryBasedActionSpaceAdapter
from decision_making.src.rl_agent.environments.sim_commands import SimCommand
from decision_making.src.rl_agent.environments.state_space.common.data_objects import EgoCentricState
from decision_making.src.rl_agent.global_types import LateralDirection


class EmergencyBrakeActionSpaceAdapter(TrajectoryBasedActionSpaceAdapter, metaclass=ABCMeta):
    RECIPE_CLS = ConstantAccelerationRLActionRecipe

    def __init__(self, action_space_params) -> None:
        """
        Abstract adapter for emergency brake action that has constant acceleration longitudinally (lateral motion not
        implemented). For jerk penalty, it approximates momentary jerk applied to switch to constant braking accel.
        """
        super().__init__(action_space_params,
                         [ConstantAccelerationRLActionRecipe(LateralDirection.SAME, decel)
                          for decel in action_space_params['DECELERATIONS']])

        self._decel_map = action_space_params['DECELERATIONS']
        self._dt = action_space_params['DECEL_BUILDUP_DT']

    def _specify_longitude(self, action_recipes: List[RECIPE_CLS],
                           projected_ego_fstates: FrenetStates2D) -> (np.ndarray, np.ndarray):
        """
        Generates longitudinal polynomial coefficients and terminal time for emergency actions as constant acceleration
        extracted from the deceleration map according to the action recipe aggressiveness level (allows the creation of
        different emergency brakes with different decel rates).
        :param action_recipes: list of N action recipes to specify
        :param projected_ego_fstates: the ego fstates for each recipe projected onto its corresponding target lane frame
        :return: 2D numpy array (shape Nx6) of longitudinal polynomial coefficients, 1D numpy array of terminal times
        """
        sx_0, sv_0, sa_0, *_ = projected_ego_fstates.transpose()
        action_max_decel = np.array([recipe.acceleration for recipe in action_recipes])

        # Generate 2D trajectory polynomials
        zeros = np.zeros_like(action_max_decel)
        T_s, sv_T, sx_T = sv_0 / np.abs(action_max_decel), zeros, sx_0 + sv_0 ** 2 / (2 * np.abs(action_max_decel))
        poly_coefs_s = np.c_[zeros, zeros, zeros, action_max_decel / 2, sv_0, sx_0]

        return poly_coefs_s, T_s

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
        traj_vels, traj_accs, traj_lat_pos = trajectory[:self._params["SIMS_PER_STEP"] + 1, [FS_SV, FS_SA, FS_DX]].T

        commands = [SimCommand(vel, 0) for vel in traj_vels[1:]]

        projected_ego_fstate = state.ego_fstate_on_adj_lanes[action_spec.relative_lane]
        commands[0] = SimCommand(commands[0].target_speed, traj_lat_pos[-1] - projected_ego_fstate[FS_DX])

        # this part approximates a momentary jerk applied over dt[sec] to reach max_decel
        max_decel = action_recipe.acceleration
        a_0, v_0 = state.ego_state.fstate[[FS_SA, FS_SV]]
        momentary_jerk = (max_decel - a_0) / self._dt if a_0 > max_decel and v_0 > 0 else 0

        info = {"traj_vels": traj_vels[1:],
                "traj_accs": traj_accs[1:],
                "action_spec": action_spec,
                "action_not_none_nor_short": action_spec is not None and action_spec.t > TRAJECTORY_TIME_RESOLUTION,
                "action_spec_none": action_spec is None,
                "intermediate_jerk": np.zeros(len(relative_time_points)-1),
                "knot_jerk_diff": momentary_jerk - (state.ego_jerk or 0),
                "sum_sq_lon_acc": samplable_trajectory.get_lon_sq_acc_at(np.array([0, 1])),
                "sum_sq_lon_jerk": self._dt * momentary_jerk ** 2,
                "sum_sq_lat_jerk": samplable_trajectory.get_lat_sq_jerk_at(np.array([0, 1]))}

        return commands, info


class EmergencyBrakeForCommitLaneChangesActionSpaceAdapter(CommitLaneChangeMixin, EmergencyBrakeActionSpaceAdapter):
    RECIPE_CLS = EmergencyBrakeActionSpaceAdapter.RECIPE_CLS


class EmergencyBrakeForNegotiationActionSpaceAdapter(NegotiateLaneChangeMixin, EmergencyBrakeActionSpaceAdapter):
    RECIPE_CLS = EmergencyBrakeActionSpaceAdapter.RECIPE_CLS
