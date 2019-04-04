import numpy as np
from logging import Logger
from sklearn.utils.extmath import cartesian
from typing import Optional, List, Type

import rte.python.profiler as prof
from decision_making.src.global_constants import BP_ACTION_T_LIMITS, SPECIFICATION_HEADWAY, \
    BP_JERK_S_JERK_D_TIME_WEIGHTS, LONGITUDINAL_SPECIFY_MARGIN_FROM_OBJECT
from decision_making.src.planning.behavioral.action_space.action_space import ActionSpace
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionSpec, DynamicActionRecipe, \
    ActionType, RelativeLongitudinalPosition
from decision_making.src.planning.behavioral.data_objects import RelativeLane, AggressivenessLevel
from decision_making.src.planning.behavioral.filtering.recipe_filtering import RecipeFiltering
from decision_making.src.planning.types import LIMIT_MAX, FS_SV, FS_SX, FS_SA, FS_DA, FS_DV, FS_DX
from decision_making.src.planning.utils.math_utils import Math
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D
from decision_making.src.prediction.ego_aware_prediction.ego_aware_predictor import EgoAwarePredictor


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
        # pick ego initial fstates projected on all target frenet_frames
        relative_lanes = np.array([recipe.relative_lane for recipe in action_recipes])
        projected_ego_fstates = np.array([behavioral_state.projected_ego_fstates[lane] for lane in relative_lanes])

        # collect targets' lengths, lane_ids and fstates
        targets = [behavioral_state.road_occupancy_grid[(action_recipe.relative_lane, action_recipe.relative_lon)][0]
                   for action_recipe in action_recipes]
        target_length = np.array([target.dynamic_object.size.length for target in targets])
        target_map_states = [target.dynamic_object.map_state for target in targets]
        # get desired terminal velocity
        v_T = np.array([map_state.lane_fstate[FS_SV] for map_state in target_map_states])

        v_0 = behavioral_state.ego_state.map_state.lane_fstate[FS_SV]
        a_0 = behavioral_state.ego_state.map_state.lane_fstate[FS_SA]

        # get relevant aggressiveness weights for all actions
        aggressiveness = np.array([action_recipe.aggressiveness.value for action_recipe in action_recipes])
        weights = BP_JERK_S_JERK_D_TIME_WEIGHTS[aggressiveness]

        # calculate initial longitudinal differences between all target objects and ego along target lanes
        longitudinal_differences = behavioral_state.calculate_longitudinal_differences(target_map_states)
        assert not np.isinf(longitudinal_differences).any()

        # margin_sign is -1 for FOLLOW_VEHICLE (behind target) and +1 for OVER_TAKE_VEHICLE (in front of target)
        margin_sign = np.array([-1 if action_recipe.action_type == ActionType.FOLLOW_VEHICLE else +1
                                for action_recipe in action_recipes])

        # here we deduct from the distance to progres: half of lengths of host and target (so we can stay in center-host
        # to center-target distance, plus another margin that will represent the stopping distance, when headway is
        # irrelevant due to 0 velocity
        ds = longitudinal_differences + margin_sign * (LONGITUDINAL_SPECIFY_MARGIN_FROM_OBJECT +
                                                       behavioral_state.ego_state.size.length / 2 + target_length / 2)

        # T_s <- find minimal non-complex local optima within the BP_ACTION_T_LIMITS bounds, otherwise <np.nan>
        cost_coeffs_s = QuinticPoly1D.time_cost_function_derivative_coefs(
            w_T=weights[:, 2], w_J=weights[:, 0], dx=ds, a_0=projected_ego_fstates[:, FS_SA],
            v_0=projected_ego_fstates[:, FS_SV], v_T=v_T, T_m=SPECIFICATION_HEADWAY)
        roots_s = Math.find_real_roots_in_limits(cost_coeffs_s, np.array([0, BP_ACTION_T_LIMITS[LIMIT_MAX]]))
        T_s = np.fmin.reduce(roots_s, axis=-1)

        # Agent is in tracking mode, meaning the required velocity change is negligible and action time is actually
        # zero. This degenerate action is valid but can't be solved analytically thus we probably got nan for T_s
        # although it should be zero. Here we can't find a local minima as the equation is close to a linear line,
        # intersecting in T=0.
        # TODO: this creates 3 actions (different aggressiveness levels) which are the same, in case of tracking mode
        T_s[QuinticPoly1D.is_tracking_mode(v_0, v_T, a_0, ds, SPECIFICATION_HEADWAY)] = 0

        # T_d <- find minimal non-complex local optima within the BP_ACTION_T_LIMITS bounds, otherwise <np.nan>
        cost_coeffs_d = QuinticPoly1D.time_cost_function_derivative_coefs(
            w_T=weights[:, 2], w_J=weights[:, 1], dx=-projected_ego_fstates[:, FS_DX],
            a_0=projected_ego_fstates[:, FS_DA], v_0=projected_ego_fstates[:, FS_DV], v_T=0, T_m=SPECIFICATION_HEADWAY)
        roots_d = Math.find_real_roots_in_limits(cost_coeffs_d, np.array([0, BP_ACTION_T_LIMITS[LIMIT_MAX]]))
        T_d = np.fmin.reduce(roots_d, axis=-1)

        # if both T_d[i] and T_s[i] are defined for i, then take maximum. otherwise leave it nan.
        T = np.maximum(T_d, T_s)

        # Calculate resulting distance from sampling the state at time T from the Quartic polynomial solution.
        # distance_s also takes into account the safe distance that depends on target vehicle velocity that we want
        # to keep from the target vehicle.
        distance_s = QuinticPoly1D.distance_profile_function(a_0=projected_ego_fstates[:, FS_SA],
                                                             v_0=projected_ego_fstates[:, FS_SV],
                                                             v_T=v_T, T=T, dx=ds,
                                                             T_m=SPECIFICATION_HEADWAY)(T)
        # Absolute longitudinal position of target
        target_s = distance_s + projected_ego_fstates[:, FS_SX]

        # lane center has latitude = 0, i.e. spec.d = 0
        action_specs = [ActionSpec(t, vt, st, 0, recipe)
                        if ~np.isnan(t) else None
                        for recipe, t, vt, st in zip(action_recipes, T, v_T, target_s)]

        return action_specs
