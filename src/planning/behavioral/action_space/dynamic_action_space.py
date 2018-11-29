from logging import Logger
from typing import Optional, List, Type

import numpy as np
from sklearn.utils.extmath import cartesian

import rte.python.profiler as prof
from decision_making.src.global_constants import BP_ACTION_T_LIMITS, SPECIFICATION_MARGIN_TIME_DELAY, \
    BP_JERK_S_JERK_D_TIME_WEIGHTS, LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT
from decision_making.src.planning.behavioral.action_space.action_space import ActionSpace
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionSpec, DynamicActionRecipe, \
    ActionType, RelativeLongitudinalPosition
from decision_making.src.planning.behavioral.data_objects import RelativeLane, AggressivenessLevel
from decision_making.src.planning.behavioral.filtering.recipe_filtering import RecipeFiltering
from decision_making.src.planning.types import LIMIT_MAX, FS_SV, FS_SX, LIMIT_MIN, FS_SA, FS_DA, FS_DV, FS_DX
from decision_making.src.planning.utils.math import Math
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D
from decision_making.src.prediction.ego_aware_prediction.ego_aware_predictor import EgoAwarePredictor
from decision_making.src.utils.map_utils import MapUtils


class DynamicActionSpace(ActionSpace):
    def __init__(self, logger: Logger, predictor: EgoAwarePredictor, filtering: RecipeFiltering):
        super().__init__(logger,
                         recipes=[DynamicActionRecipe.from_args_list(comb)
                                  for comb in cartesian([RelativeLane,
                                                         RelativeLongitudinalPosition,
                                                         [ActionType.FOLLOW_VEHICLE, ActionType.OVERTAKE_VEHICLE],
                                                         AggressivenessLevel])],
                         recipe_filtering=filtering)
        self.predictor = predictor

    @property
    def recipe_classes(self) -> List[Type]:
        """a list of Recipe classes this action space can handle with"""
        return [DynamicActionRecipe]

    @prof.ProfileFunction()
    def specify_goals(self, action_recipes: List[DynamicActionRecipe], behavioral_state: BehavioralGridState) -> \
            List[Optional[ActionSpec]]:
        """
        This method's purpose is to specify the enumerated actions (recipes) that the agent can take.
        Each semantic action (ActionRecipe) is translated into a terminal state specification (ActionSpec).
        :param action_recipes: a list of enumerated semantic actions [ActionRecipe].
        :param behavioral_state: a Frenet state of ego at initial point
        :return: semantic action specification [ActionSpec] or [None] if recipe can't be specified.
        """
        ego = behavioral_state.ego_state

        targets = [behavioral_state.road_occupancy_grid[(action_recipe.relative_lane, action_recipe.relative_lon)][0]
                   for action_recipe in action_recipes]

        relative_lanes_per_action = [recipe.relative_lane for recipe in action_recipes]
        # project ego on target lane frenet_frame
        projected_fstates = ego.project_on_adjacent_lanes()
        ego_init_fstates = np.array([projected_fstates[recipe.relative_lane] for recipe in action_recipes])

        target_length = np.array([target.dynamic_object.size.length for target in targets])
        target_fstate = np.array([target.dynamic_object.map_state.lane_fstate for target in targets])

        # get relevant aggressiveness weights for all actions
        aggressiveness = np.array([action_recipe.aggressiveness.value for action_recipe in action_recipes])
        weights = BP_JERK_S_JERK_D_TIME_WEIGHTS[aggressiveness]

        # get desired terminal velocity
        v_T = target_fstate[:, FS_SV]

        # longitudinal difference between object and ego at t=0 (positive if obj in front of ego)
        init_longitudinal_difference = target_fstate[:, FS_SX] - ego_init_fstates[:, FS_SX]
        # margin_sign is -1 for FOLLOW_VEHICLE (behind target) and +1 for OVER_TAKE_VEHICLE (in front of target)
        margin_sign = np.array([-1 if action_recipe.action_type == ActionType.FOLLOW_VEHICLE else +1
                                for action_recipe in action_recipes])

        ds = init_longitudinal_difference + margin_sign * (
            LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT + ego.size.length / 2 + target_length / 2)

        # T_s <- find minimal non-complex local optima within the BP_ACTION_T_LIMITS bounds, otherwise <np.nan>
        cost_coeffs_s = QuinticPoly1D.time_cost_function_derivative_coefs(
            w_T=weights[:, 2], w_J=weights[:, 0], dx=ds,
            a_0=ego_init_fstates[:, FS_SA], v_0=ego_init_fstates[:, FS_SV], v_T=v_T, T_m=SPECIFICATION_MARGIN_TIME_DELAY)
        roots_s = Math.find_real_roots_in_limits(cost_coeffs_s, np.array([0, BP_ACTION_T_LIMITS[LIMIT_MAX]]))
        T_s = np.fmin.reduce(roots_s, axis=-1)

        # voids (setting <np.nan>) all non-Calm actions with T_s < (minimal allowed T_s)
        # this still leaves some values of T_s which are smaller than (minimal allowed T_s) and will be replaced later
        # when setting T
        with np.errstate(invalid='ignore'):
            T_s[(T_s < BP_ACTION_T_LIMITS[LIMIT_MIN]) & (aggressiveness > AggressivenessLevel.CALM.value)] = np.nan

        # T_d <- find minimal non-complex local optima within the BP_ACTION_T_LIMITS bounds, otherwise <np.nan>
        cost_coeffs_d = QuinticPoly1D.time_cost_function_derivative_coefs(
            w_T=weights[:, 2], w_J=weights[:, 1], dx=-ego_init_fstates[:, FS_DX],
            a_0=ego_init_fstates[:, FS_DA], v_0=ego_init_fstates[:, FS_DV], v_T=0, T_m=SPECIFICATION_MARGIN_TIME_DELAY)
        roots_d = Math.find_real_roots_in_limits(cost_coeffs_d, np.array([0, BP_ACTION_T_LIMITS[LIMIT_MAX]]))
        T_d = np.fmin.reduce(roots_d, axis=-1)

        # if both T_d[i] and T_s[i] are defined for i, then take maximum. otherwise leave it nan.
        T = np.maximum(np.maximum(T_d, T_s), BP_ACTION_T_LIMITS[LIMIT_MIN])

        # Calculate resulting distance from sampling the state at time T from the Quartic polynomial solution.
        # distance_s also takes into account the safe distance that depends on target vehicle velocity that we want
        # to keep from the target vehicle.
        distance_s = QuinticPoly1D.distance_profile_function(a_0=ego_init_fstates[:, FS_SA],
                                                             v_0=ego_init_fstates[:, FS_SV],
                                                             v_T=v_T, T=T, dx=ds,
                                                             T_m=SPECIFICATION_MARGIN_TIME_DELAY)(T)
        # Absolute longitudinal position of target
        target_s = distance_s + ego_init_fstates[:, FS_SX]

        lane_id = ego.map_state.lane_id
        right_lanes = MapUtils.get_adjacent_lanes(lane_id, RelativeLane.RIGHT_LANE)
        left_lanes = MapUtils.get_adjacent_lanes(lane_id, RelativeLane.LEFT_LANE)
        adjacent_lanes = {RelativeLane.RIGHT_LANE: right_lanes[0] if len(right_lanes) > 0 else None,
                          RelativeLane.SAME_LANE: lane_id,
                          RelativeLane.LEFT_LANE: left_lanes[0] if len(left_lanes) > 0 else None}

        # lane center has latitude = 0, i.e. spec.d = 0
        action_specs = [ActionSpec(t, v_T[i], target_s[i], 0, adjacent_lanes[relative_lanes_per_action[i]])
                        if ~np.isnan(t) else None
                        for i, t in enumerate(T)]

        return action_specs
