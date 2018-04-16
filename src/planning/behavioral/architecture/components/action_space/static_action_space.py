from typing import Optional

import numpy as np
from sklearn.utils.extmath import cartesian

from decision_making.src.global_constants import BP_ACTION_T_LIMITS, BP_ACTION_T_RES
from decision_making.src.planning.behavioral.architecture.components.action_space.action_space import ActionSpace
from decision_making.src.planning.behavioral.architecture.components.filtering import recipe_filter_bank
from decision_making.src.planning.behavioral.architecture.components.filtering.recipe_filtering import RecipeFiltering
from decision_making.src.planning.behavioral.architecture.constants import VELOCITY_STEP, MAX_VELOCITY, MIN_VELOCITY
from decision_making.src.planning.behavioral.architecture.data_objects import ActionSpec, StaticActionRecipe
from decision_making.src.planning.behavioral.architecture.data_objects import RelativeLane, AggressivenessLevel
from decision_making.src.planning.behavioral.architecture.semantic_behavioral_grid_state import \
    SemanticBehavioralGridState
from decision_making.src.planning.trajectory.optimal_control.optimal_control_utils import QuinticPoly1D, QuarticPoly1D
from decision_making.src.planning.trajectory.optimal_control.werling_planner import SamplableWerlingTrajectory
from decision_making.src.planning.types import LIMIT_MAX, LIMIT_MIN
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.planning.utils.math import Math
from mapping.src.service.map_service import MapService


class StaticActionSpace(ActionSpace):
    def __init__(self, logger):
        self._velocity_grid = np.arange(MIN_VELOCITY, MAX_VELOCITY + np.finfo(np.float16).eps, VELOCITY_STEP)
        super().__init__(logger,
                         recipes=[StaticActionRecipe.from_args_list(comb)
                                  for comb in cartesian([RelativeLane, self._velocity_grid, AggressivenessLevel])],
                         recipe_filtering=RecipeFiltering(recipe_filter_bank.static_filters))

    def specify_goal(self, action_recipe: StaticActionRecipe, behavioral_state: SemanticBehavioralGridState) -> \
            Optional[ActionSpec]:
        ego = behavioral_state.ego_state
        ego_init_cstate = np.array([ego.x, ego.y, ego.yaw, ego.v_x, ego.acceleration_lon, ego.curvature])
        road_id = ego.road_localization.road_id
        road_points = MapService.get_instance()._shift_road_points_to_latitude(road_id, 0.0)  # TODO: use nav_plan
        road_frenet = FrenetSerret2DFrame(road_points)
        ego_init_fstate = road_frenet.cstate_to_fstate(ego_init_cstate)

        road_lane_latitudes = MapService.get_instance().get_center_lanes_latitudes(road_id)
        desired_lane = ego.road_localization.lane_num + action_recipe.relative_lane.value
        desired_center_lane_latitude = road_lane_latitudes[desired_lane]
        T_vals = np.arange(BP_ACTION_T_LIMITS[LIMIT_MIN], BP_ACTION_T_LIMITS[LIMIT_MAX] + np.finfo(np.float16).eps,
                           BP_ACTION_T_RES)

        constraints_s = ActionSpace.define_lon_constraints(len(T_vals), ego_init_fstate, 0.0, action_recipe.velocity)
        constraints_d = ActionSpace.define_lat_constraints(len(T_vals), ego_init_fstate, desired_center_lane_latitude)

        A_inv_s = np.linalg.inv(QuarticPoly1D.time_constraints_tensor(T_vals))
        A_inv_d = np.linalg.inv(QuinticPoly1D.time_constraints_tensor(T_vals))

        # solve for s(t) and d(t)
        poly_coefs_s = QuarticPoly1D.zip_solve(A_inv_s, constraints_s)
        poly_coefs_d = QuinticPoly1D.zip_solve(A_inv_d, constraints_d)

        optimum_time_idx, optimum_time_satisfies_constraints = ActionSpace.find_optimum_planning_time(T_vals,
                                                                                                      poly_coefs_s,
                                                                                                      QuarticPoly1D,
                                                                                                      poly_coefs_d,
                                                                                                      QuinticPoly1D,
                                                                                                      action_recipe.aggressiveness)

        if not optimum_time_satisfies_constraints:
            # self.logger.debug("Can\'t specify Recipe %s given ego state %s ", str(action_recipe), str(ego))
            return None

        # Note: We create the samplable trajectory as a reference trajectory of the current action.from
        # We assume correctness only of the longitudinal axis, and set T_d to be equal to T_s.
        samplable_trajectory = SamplableWerlingTrajectory(timestamp_in_sec=ego.timestamp_in_sec,
                                                          T_s=T_vals[optimum_time_idx],
                                                          T_d=T_vals[optimum_time_idx],
                                                          frenet_frame=road_frenet,
                                                          poly_s_coefs=poly_coefs_s[optimum_time_idx],
                                                          poly_d_coefs=poly_coefs_d[optimum_time_idx])

        target_s = np.polyval(poly_coefs_s[optimum_time_idx], T_vals[optimum_time_idx])

        return ActionSpec(t=T_vals[optimum_time_idx], v=constraints_s[optimum_time_idx, 3],
                          s=float(target_s),
                          d=constraints_d[optimum_time_idx, 3],
                          samplable_trajectory=samplable_trajectory)
