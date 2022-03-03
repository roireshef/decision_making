from abc import abstractmethod
from typing import List

import numpy as np
from decision_making.src.global_constants import BP_JERK_S_JERK_D_TIME_WEIGHTS
from decision_making.src.planning.types import FrenetStates2D, FS_SV, FS_SX
from decision_making.src.planning.utils.math_utils import Math
from decision_making.src.planning.utils.optimal_control.poly1d import QuarticPoly1D
from decision_making.src.rl_agent.environments.action_space.common.data_objects import RLActionRecipe
from decision_making.src.rl_agent.utils.frenet_utils import FrenetUtils


class LongitudinalMixin:
    """ Mixin for TrajectoryBasedActionSpaceAdapter implementations that uses polynomials for longitudinal motion """
    @abstractmethod
    def _specify_longitude(self, action_recipes: List[RLActionRecipe], projected_ego_fstates: FrenetStates2D) -> \
            (np.ndarray, np.ndarray):
        """
        a method that takes a list of recipes and the ego state and generates longitudinal motion profiles with their
        time horizons.
        :param action_recipes: list of N action recipes to specify
        :param projected_ego_fstates: the ego fstates for each recipe projected onto its corresponding target lane frame
        :return: 2D numpy array (shape Nx6) of longitudinal polynomial coefficients, 1D numpy array of terminal times
        """
        pass


class TerminalVelocityMixin(LongitudinalMixin):
    """ Longitudinal mixin for Jerk-minimizing (Werling-based) method to create the longitudinal motion
     polynomial based on target terminal velocity and aggressiveness levels """

    def _specify_longitude(self, action_recipes: List[RLActionRecipe], projected_ego_fstates: FrenetStates2D) -> \
            (np.ndarray, np.ndarray):
        """
        Takes action recipes and ego state projection (on the correct target lanes) and specifies the longitudinal
        polynomial and terminal time (horizon) of motion for each recipe, by solving boundary conditions for the
        Jerk-optimal case.
        :param action_recipes: list of N action recipes to specify
        :param projected_ego_fstates: the ego fstates for each recipe projected onto its corresponding target lane frame
        :return: 2D numpy array (shape Nx6) of longitudinal polynomial coefficients, 1D numpy array of terminal times
        """
        # get relevant aggressiveness weights and target velocity for all actions
        aggressiveness, sv_T = np.array([[action_recipe.aggressiveness.value, action_recipe.velocity]
                                         for action_recipe in action_recipes]).transpose()
        weights = BP_JERK_S_JERK_D_TIME_WEIGHTS[aggressiveness.astype(int)]

        sx_0, sv_0, sa_0, *_ = projected_ego_fstates.transpose()
        w_T, w_J = weights[:, 2], weights[:, 0]

        # T_s <- find minimal non-complex local optima within the BP_ACTION_T_LIMITS bounds, otherwise <np.nan>
        cost_coefs_s = QuarticPoly1D.time_cost_function_derivative_coefs(a_0=sa_0, v_0=sv_0, v_T=sv_T, w_T=w_T, w_J=w_J)
        roots_s = Math.find_real_roots_in_limits(cost_coefs_s, np.array([.0, self._params['ACTION_MAX_TIME_HORIZON']]))
        T_s = np.fmin.reduce(roots_s, axis=-1)
        T_s[QuarticPoly1D.is_tracking_mode(sv_0, sv_T, sa_0)] = 0

        # Calculate resulting distance from sampling the state at time T from the Quartic polynomial solution
        # QuarticPoly1D.distance_profile_function returns NaN when T_s == 0, so in that case we override result with 0
        distance_s = np.nan_to_num(QuarticPoly1D.distance_profile_function(a_0=sa_0, v_0=sv_0, v_T=sv_T, T=T_s)(T_s))
        sx_T = distance_s + sx_0

        # Generate 2D trajectory polynomials
        poly_coefs_s = FrenetUtils.generate_polynomials_1d(sx_0, sv_0, sa_0, sx_T, sv_T, T_s)

        return poly_coefs_s, T_s


class AccelerationCommandsMixin(LongitudinalMixin):
    """ Longitudinal mixin that applies acceleration/deceleration for some horizon """

    def _specify_longitude(self, action_recipes: List[RLActionRecipe], projected_ego_fstates: FrenetStates2D) -> \
            (np.ndarray, np.ndarray):
        poly_coefs_s = np.array([[0, 0, 0, recipe.acceleration / 2, fstate[FS_SV], fstate[FS_SX]]
                                 for recipe, fstate in zip(action_recipes, projected_ego_fstates)])

        return poly_coefs_s, np.full(len(action_recipes), self._params["ACTION_HORIZON"])
