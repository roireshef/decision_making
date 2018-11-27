from typing import Optional, List, Type, Dict

import numpy as np
from sklearn.utils.extmath import cartesian

import rte.python.profiler as prof
from decision_making.src.global_constants import BP_ACTION_T_LIMITS, BP_JERK_S_JERK_D_TIME_WEIGHTS, VELOCITY_LIMITS
from decision_making.src.global_constants import VELOCITY_STEP
from decision_making.src.planning.behavioral.action_space.action_space import ActionSpace
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionSpec, StaticActionRecipe
from decision_making.src.planning.behavioral.data_objects import RelativeLane, AggressivenessLevel
from decision_making.src.planning.behavioral.filtering.recipe_filtering import RecipeFiltering
from decision_making.src.planning.types import LIMIT_MAX, LIMIT_MIN, FS_SV, FS_SA, FS_DX, FS_DA, FS_DV, FS_SX
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GeneralizedFrenetSerretFrame
from decision_making.src.planning.utils.math import Math
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D, QuarticPoly1D
from decision_making.src.utils.map_utils import MapUtils


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

    @prof.ProfileFunction()
    def specify_goals(self, action_recipes: List[StaticActionRecipe], behavioral_state: BehavioralGridState,
                      unified_frames: Dict[RelativeLane, GeneralizedFrenetSerretFrame]) -> \
            List[Optional[ActionSpec]]:
        """
        This method's purpose is to specify the enumerated actions (recipes) that the agent can take.
        Each semantic action (ActionRecipe) is translated into a terminal state specification (ActionSpec).
        :param action_recipes: a list of enumerated semantic actions [ActionRecipe].
        :param behavioral_state: a Frenet state of ego at initial point
        :return: semantic action specification [ActionSpec] or [None] if recipe can't be specified.
        """
        ego = behavioral_state.ego_state

        # get the relevant desired center lane latitude (from road's RHS)
        relative_lanes = [action_recipe.relative_lane for action_recipe in action_recipes]
        # project ego on target lane frenet_frame
        ego_init_fstates = np.array(ego.project_on_relative_lanes(relative_lanes))

        # get relevant aggressiveness weights for all actions
        aggressiveness = np.array([action_recipe.aggressiveness.value for action_recipe in action_recipes])
        weights = BP_JERK_S_JERK_D_TIME_WEIGHTS[aggressiveness]

        # get desired terminal velocity
        v_T = np.array([action_recipe.velocity for action_recipe in action_recipes])

        # T_s <- find minimal non-complex local optima within the BP_ACTION_T_LIMITS bounds, otherwise <np.nan>
        cost_coeffs_s = QuarticPoly1D.time_cost_function_derivative_coefs(
            w_T=weights[:, 2], w_J=weights[:, 0], a_0=ego_init_fstates[:, FS_SA], v_0=ego_init_fstates[:, FS_SV], v_T=v_T)
        roots_s = Math.find_real_roots_in_limits(cost_coeffs_s, np.array([0, BP_ACTION_T_LIMITS[LIMIT_MAX]]))
        T_s = np.fmin.reduce(roots_s, axis=-1)

        # voids (setting <np.nan>) all non-Calm actions with T_s < (minimal allowed T_s)
        # this still leaves some values of T_s which are smaller than (minimal allowed T_s) and will be replaced later
        # when setting T
        with np.errstate(invalid='ignore'):
            T_s[(T_s < BP_ACTION_T_LIMITS[LIMIT_MIN]) & (aggressiveness > AggressivenessLevel.CALM.value)] = np.nan

        # T_d <- find minimal non-complex local optima within the BP_ACTION_T_LIMITS bounds, otherwise <np.nan>
        cost_coeffs_d = QuinticPoly1D.time_cost_function_derivative_coefs(
            w_T=weights[:, 2], w_J=weights[:, 1], a_0=ego_init_fstates[:, FS_DA], v_0=ego_init_fstates[:, FS_DV], v_T=0,
            dx=-ego_init_fstates[:, FS_DX], T_m=0)
        roots_d = Math.find_real_roots_in_limits(cost_coeffs_d, np.array([0, BP_ACTION_T_LIMITS[LIMIT_MAX]]))
        T_d = np.fmin.reduce(roots_d, axis=-1)

        # if both T_d[i] and T_s[i] are defined for i, then take maximum. otherwise leave it nan.
        T = np.maximum(np.maximum(T_d, T_s), BP_ACTION_T_LIMITS[LIMIT_MIN])

        # Calculate resulting distance from sampling the state at time T from the Quartic polynomial solution
        distance_s = QuarticPoly1D.distance_profile_function(a_0=ego_init_fstates[:, FS_SA],
                                                             v_0=ego_init_fstates[:, FS_SV], v_T=v_T, T=T)(T)
        # Absolute longitudinal position of target
        target_s = distance_s + ego_init_fstates[:, FS_SX]

        lane_id = ego.map_state.lane_id
        right_lanes = MapUtils.get_adjacent_lanes(lane_id, RelativeLane.RIGHT_LANE)
        left_lanes = MapUtils.get_adjacent_lanes(lane_id, RelativeLane.LEFT_LANE)
        adjacent_lanes = {RelativeLane.RIGHT_LANE: right_lanes[0] if len(right_lanes) > 0 else None,
                          RelativeLane.SAME_LANE: lane_id,
                          RelativeLane.LEFT_LANE: left_lanes[0] if len(left_lanes) > 0 else None}

        action_specs = [ActionSpec(t, v_T[i], target_s[i], 0, adjacent_lanes[relative_lanes[i]])
                        if ~np.isnan(t) else None
                        for i, t in enumerate(T)]

        return action_specs
