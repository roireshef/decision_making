from typing import Optional, List, Type

import numpy as np
from sklearn.utils.extmath import cartesian

from decision_making.src.global_constants import BP_ACTION_T_LIMITS, BP_JERK_S_JERK_D_TIME_WEIGHTS, \
    SAFE_DIST_TIME_DELAY, VELOCITY_LIMITS
from decision_making.src.planning.behavioral.action_space.action_space import ActionSpace
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.global_constants import VELOCITY_STEP
from decision_making.src.planning.behavioral.data_objects import ActionSpec, StaticActionRecipe
from decision_making.src.planning.behavioral.data_objects import RelativeLane, AggressivenessLevel
from decision_making.src.planning.behavioral.filtering.recipe_filtering import RecipeFiltering
from decision_making.src.planning.types import LIMIT_MAX, LIMIT_MIN, FS_SV, FS_SA, FS_DX, FS_DA, FS_DV, FS_SX
from decision_making.src.planning.utils.map_utils import MapUtils
from decision_making.src.planning.utils.math import Math
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D, QuarticPoly1D
from mapping.src.service.map_service import MapService


class StaticActionSpace(ActionSpace):
    def __init__(self, logger, filtering: RecipeFiltering):
        self._velocity_grid = np.arange(VELOCITY_LIMITS[LIMIT_MIN],
                                        VELOCITY_LIMITS[LIMIT_MAX] + np.finfo(np.float16).eps,
                                        VELOCITY_STEP)
        super().__init__(logger,
                         recipes=[StaticActionRecipe.from_args_list(comb)
                                  for comb in cartesian([RelativeLane, self._velocity_grid, AggressivenessLevel])],
                         recipe_filtering=filtering)

    @property
    def recipe_classes(self) -> List[Type]:
        """a list of Recipe classes this action space can handle with"""
        return [StaticActionRecipe]

    def specify_goals(self, action_recipes: List[StaticActionRecipe], behavioral_state: BehavioralGridState) -> \
            List[Optional[ActionSpec]]:
        """
        This method's purpose is to specify the enumerated actions (recipes) that the agent can take.
        Each semantic action (ActionRecipe) is translated into a terminal state specification (ActionSpec).
        :param action_recipes: an enumerated semantic action [ActionRecipe].
        :param behavioral_state: a Frenet state of ego at initial point
        :return: semantic action specification [ActionSpec] or [None] if recipe can't be specified.
        """
        ego = behavioral_state.ego_state
        ego_init_fstate = ego.map_state

        # get the relevant desired center lane latitude (from road's RHS)
        road_id = ego.map_state.road_id
        # TODO: create method get_center_lanes_latitudes
        road_lane_latitudes = MapService.get_instance().get_center_lanes_latitudes(road_id)
        relative_lane = np.array([action_recipe.relative_lane.value for action_recipe in action_recipes])
        ego_lane_num = MapService.get_instance().get_lane(ego.map_state.lane_id).ordinal
        desired_lane = ego_lane_num + relative_lane
        desired_center_lane_latitude = road_lane_latitudes[desired_lane]

        # get relevant aggressiveness weights for all actions
        aggressiveness = np.array([action_recipe.aggressiveness.value for action_recipe in action_recipes])
        weights = BP_JERK_S_JERK_D_TIME_WEIGHTS[aggressiveness]

        # get desired terminal velocity
        v_T = np.array([action_recipe.velocity for action_recipe in action_recipes])

        # T_s <- find minimal non-complex local optima within the BP_ACTION_T_LIMITS bounds, otherwise <np.nan>
        cost_coeffs_s = QuarticPoly1D.time_cost_function_derivative_coefs(
            w_T=weights[:, 2], w_J=weights[:, 0], a_0=ego_init_fstate[FS_SA], v_0=ego_init_fstate[FS_SV], v_T=v_T)
        roots_s = Math.find_real_roots_in_limits(cost_coeffs_s, np.array([0, BP_ACTION_T_LIMITS[LIMIT_MAX]]))
        T_s = np.fmin.reduce(roots_s, axis=-1)

        # voids (setting <np.nan>) all non-Calm actions with T_s < (minimal allowed T_s)
        # this still leaves some values of T_s which are smaller than (minimal allowed T_s) and will be replaced later
        # when setting T
        T_s[(T_s < BP_ACTION_T_LIMITS[LIMIT_MIN]) & (aggressiveness > AggressivenessLevel.CALM.value)] = np.nan

        # latitudinal difference to target
        latitudinal_difference = desired_center_lane_latitude - ego_init_fstate[FS_DX]

        # T_d <- find minimal non-complex local optima within the BP_ACTION_T_LIMITS bounds, otherwise <np.nan>
        cost_coeffs_d = QuinticPoly1D.time_cost_function_derivative_coefs(
            w_T=weights[:, 2], w_J=weights[:, 1], a_0=ego_init_fstate[FS_DA], v_0=ego_init_fstate[FS_DV], v_T=0,
            dx=latitudinal_difference, T_m=SAFE_DIST_TIME_DELAY)
        roots_d = Math.find_real_roots_in_limits(cost_coeffs_d, np.array([0, BP_ACTION_T_LIMITS[LIMIT_MAX]]))
        T_d = np.fmin.reduce(roots_d, axis=-1)

        # if both T_d[i] and T_s[i] are defined for i, then take maximum. otherwise leave it nan.
        T = np.maximum(np.maximum(T_d, T_s), BP_ACTION_T_LIMITS[LIMIT_MIN])

        # Calculate resulting distance from sampling the state at time T from the Quartic polynomial solution
        distance_s = QuarticPoly1D.distance_profile_function(a_0=ego_init_fstate[FS_SA], v_0=ego_init_fstate[FS_SV],
                                                             v_T=v_T, T=T)(T)
        target_s = distance_s + ego_init_fstate[FS_SX]

        action_specs = [ActionSpec(t, v_T[i], target_s[i], desired_center_lane_latitude[i])
                        if ~np.isnan(t) else None
                        for i, t in enumerate(T)]

        return action_specs
