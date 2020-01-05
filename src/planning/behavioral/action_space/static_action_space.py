import numpy as np

import rte.python.profiler as prof
from decision_making.src.global_constants import BP_ACTION_T_LIMITS, BP_JERK_S_JERK_D_TIME_WEIGHTS, VELOCITY_LIMITS, \
    EPS, LANE_CHANGE_TIME_COMPLETION_TARGET, MIN_LANE_CHANGE_ACTION_TIME
from decision_making.src.global_constants import VELOCITY_STEP
from decision_making.src.planning.behavioral.action_space.action_space import ActionSpace
from decision_making.src.planning.behavioral.state.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.state.lane_change_state import LaneChangeStatus
from decision_making.src.planning.behavioral.data_objects import ActionSpec, StaticActionRecipe
from decision_making.src.planning.behavioral.data_objects import RelativeLane, AggressivenessLevel
from decision_making.src.planning.behavioral.filtering.recipe_filtering import RecipeFiltering
from decision_making.src.planning.types import LIMIT_MAX, LIMIT_MIN, FS_SV, FS_SA, FS_DX, FS_DA, FS_DV, FS_SX
from decision_making.src.planning.utils.kinematics_utils import KinematicUtils
from decision_making.src.planning.utils.math_utils import Math
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D, QuarticPoly1D
from sklearn.utils.extmath import cartesian
from typing import Optional, List, Type


class StaticActionSpace(ActionSpace):
    def __init__(self, logger, filtering: RecipeFiltering, speed_limit: float = None):
        self._velocity_grid = np.arange(VELOCITY_LIMITS[LIMIT_MIN],
                                        VELOCITY_LIMITS[LIMIT_MAX] + EPS,
                                        VELOCITY_STEP)
        if speed_limit is not None:
            self._velocity_grid = np.append(self._velocity_grid, speed_limit)

        super().__init__(logger,
                         recipes=[StaticActionRecipe.from_args_list(comb)
                                  for comb in cartesian([RelativeLane, self._velocity_grid, AggressivenessLevel])],
                         recipe_filtering=filtering)

    @property
    def recipe_classes(self) -> List[Type]:
        """a list of Recipe classes this action space can handle with"""
        return [StaticActionRecipe]

    @prof.ProfileFunction()
    def specify_goals(self, action_recipes: List[StaticActionRecipe], behavioral_state: BehavioralGridState) -> \
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

        # get relevant aggressiveness weights for all actions
        aggressiveness = np.array([action_recipe.aggressiveness.value for action_recipe in action_recipes])
        weights = BP_JERK_S_JERK_D_TIME_WEIGHTS[aggressiveness]

        # get desired terminal velocity
        v_T = np.array([action_recipe.velocity for action_recipe in action_recipes])

        # T_s <- find minimal non-complex local optima within the BP_ACTION_T_LIMITS bounds, otherwise <np.nan>
        cost_coeffs_s = QuarticPoly1D.time_cost_function_derivative_coefs(
            w_T=weights[:, 2], w_J=weights[:, 0], a_0=projected_ego_fstates[:, FS_SA], v_0=projected_ego_fstates[:, FS_SV], v_T=v_T)
        roots_s = Math.find_real_roots_in_limits(cost_coeffs_s, BP_ACTION_T_LIMITS)
        # fmin.reduce() is used instead of amin() as it ignores Nan values if it can
        T_s = np.fmin.reduce(roots_s, axis=-1)

        # Agent is in tracking mode, meaning the required velocity change is negligible and action time is actually
        # zero. This degenerate action is valid but can't be solved analytically thus we probably got nan for T_s
        # although it should be zero. Here we can't find a local minima as the equation is close to a linear line,
        # intersecting in T=0.
        # TODO: this creates 3 actions (different aggressiveness levels) which are the same, in case of tracking mode
        v_0 = behavioral_state.ego_state.map_state.lane_fstate[FS_SV]
        a_0 = behavioral_state.ego_state.map_state.lane_fstate[FS_SA]
        T_s[QuarticPoly1D.is_tracking_mode(v_0, v_T, a_0)] = 0

        # T_d <- find minimal non-complex local optima within the BP_ACTION_T_LIMITS bounds, otherwise <np.nan>
        T_d = KinematicUtils.specify_lateral_planning_time(
            projected_ego_fstates[:, FS_DA], projected_ego_fstates[:, FS_DV], -projected_ego_fstates[:, FS_DX])

        # if both T_d[i] and T_s[i] are defined for i, then take maximum. otherwise leave it nan.
        T = np.maximum(T_d, T_s)

        # Override action times if a lane change is being performed
        if behavioral_state.lane_change_state.status in [LaneChangeStatus.ANALYZING_SAFETY, LaneChangeStatus.ACTIVE_IN_SOURCE_LANE]:
            action_recipe_relative_lanes = [recipe.relative_lane for recipe in action_recipes]
            lane_change_mask = behavioral_state.lane_change_state.get_lane_change_mask(action_recipe_relative_lanes,
                                                                                       behavioral_state.extended_lane_frames)

            # Override mask values if T is nan for that recipe
            lane_change_mask = [mask if ~np.isnan(T[i]) else False for i, mask in enumerate(lane_change_mask)]

            if behavioral_state.lane_change_state.status == LaneChangeStatus.ANALYZING_SAFETY:
                # This will be reached before a lane change has begun
                T[lane_change_mask] = LANE_CHANGE_TIME_COMPLETION_TARGET
            elif behavioral_state.lane_change_state.status == LaneChangeStatus.ACTIVE_IN_SOURCE_LANE:
                T[lane_change_mask] = max(MIN_LANE_CHANGE_ACTION_TIME,
                                          LANE_CHANGE_TIME_COMPLETION_TARGET
                                          + behavioral_state.lane_change_state.lane_change_start_time
                                          - behavioral_state.ego_state.timestamp_in_sec)

        elif behavioral_state.lane_change_state.status in [LaneChangeStatus.ACTIVE_IN_TARGET_LANE]:
            # If no lane change recipes passed the filters but a lane change is currently active, then override the goal time for the
            # same lane actions. These are the actions that will be used to complete a lane change.
            same_lane_mask = [recipe.relative_lane == RelativeLane.SAME_LANE if ~np.isnan(T[i]) else False
                              for i, recipe in enumerate(action_recipes)]
            T[same_lane_mask] = max(MIN_LANE_CHANGE_ACTION_TIME,
                                    LANE_CHANGE_TIME_COMPLETION_TARGET
                                    + behavioral_state.lane_change_state.lane_change_start_time
                                    - behavioral_state.ego_state.timestamp_in_sec)

        # Calculate resulting distance from sampling the state at time T from the Quartic polynomial solution
        distance_s = QuarticPoly1D.distance_profile_function(a_0=projected_ego_fstates[:, FS_SA],
                                                             v_0=projected_ego_fstates[:, FS_SV], v_T=v_T, T=T)(T)
        # Absolute longitudinal position of target
        target_s = distance_s + projected_ego_fstates[:, FS_SX]

        # lane center has latitude = 0, i.e. spec.d = 0
        action_specs = [ActionSpec(t, vt, st, 0, recipe)
                        if ~np.isnan(t) else None
                        for recipe, t, vt, st in zip(action_recipes, T, v_T, target_s)]

        return action_specs
