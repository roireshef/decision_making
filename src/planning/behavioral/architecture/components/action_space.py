from abc import abstractmethod
from typing import List, Optional

import numpy as np
from sklearn.utils.extmath import cartesian
from logging import Logger

from decision_making.src.global_constants import BP_ACTION_T_LIMITS, BP_ACTION_T_RES, SAFE_DIST_TIME_DELAY, \
    LON_ACC_LIMITS, LAT_ACC_LIMITS, VELOCITY_LIMITS, BP_JERK_S_JERK_D_TIME_WEIGHTS
from decision_making.src.exceptions import raises
from decision_making.src.planning.behavioral.architecture.components.filtering import recipe_filter_methods
from decision_making.src.planning.behavioral.architecture.constants import VELOCITY_STEP, MAX_VELOCITY, MIN_VELOCITY
from decision_making.src.planning.behavioral.architecture.data_objects import ActionSpec, StaticActionRecipe, \
    DynamicActionRecipe, ActionType, RelativeLongitudinalPosition
from decision_making.src.planning.behavioral.architecture.data_objects import RelativeLane, AggressivenessLevel, \
    ActionRecipe
from decision_making.src.planning.behavioral.architecture.components.filtering.recipe_filtering import RecipeFiltering, RecipeFilter
from decision_making.src.planning.behavioral.architecture.semantic_behavioral_grid_state import \
    SemanticBehavioralGridState
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState
from decision_making.src.planning.utils.math import Math
from decision_making.src.planning.behavioral.policies.semantic_actions_utils import SemanticActionsUtils
from decision_making.src.planning.trajectory.optimal_control.optimal_control_utils import QuinticPoly1D, QuarticPoly1D, \
    Poly1D
from decision_making.src.planning.trajectory.optimal_control.werling_planner import SamplableWerlingTrajectory
from decision_making.src.planning.types import FP_SX, LIMIT_MAX, FS_SV, FS_SX, LIMIT_MIN, FS_SA, FS_DX, FS_DV, FS_DA, \
    FrenetState2D
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.prediction.predictor import Predictor
from mapping.src.service.map_service import MapService


class ActionSpace:
    def __init__(self, logger: Logger):
        self._recipes = []
        self.logger = logger
        self.recipe_filtering = RecipeFiltering()
        self._init_filters()

    @property
    def action_space_size(self) -> int:
        return len(self._recipes)

    @property
    def recipes(self) -> List[ActionRecipe]:
        return self._recipes

    def filter_recipe(self, recipe: ActionRecipe, behavioral_state: BehavioralState) -> bool:
        return self.recipe_filtering.filter_recipe(recipe, behavioral_state)

    def filter_recipes(self, action_recipes: List[ActionRecipe], behavioral_state: BehavioralState):
        """"""
        return [self.filter_recipe(action_recipe, behavioral_state) for action_recipe in action_recipes]

    def _init_filters(self):
        """
        Uses method add_filter of self.recipe_filtering to add RecipeFilter objects which will be applied on recipes
        when methods self.filter_recipe or self.filter_recipes will be used.
        """
        pass

    @abstractmethod
    def specify_goal(self, action_recipe: ActionRecipe, behavioral_state: BehavioralState) -> Optional[ActionSpec]:
        """
        This method's purpose is to specify the enumerated actions that the agent can take.
        Each semantic action (action_recipe) is translated to a trajectory of the agent.
        The trajectory specification is created towards a target object in given cell in case of Dynamic action,
        and towards a certain lane in case of Static action, considering ego state.
        Internally, the reference route here is the RHS of the road, and the ActionSpec is specified with respect to it.
        :param action_recipe: an enumerated semantic action [ActionRecipe].
        :param behavioral_state: Frenet state of ego at initial point
        :return: semantic action specification [ActionSpec] or [None] if recipe can't be specified.
        """
        pass

    @staticmethod
    def find_optimum_planning_time(T_vals: np.ndarray, poly_coefs_s: np.ndarray, poly_lib_s: Poly1D,
                                   poly_coefs_d: np.ndarray, poly_lib_d: Poly1D, agg_level: AggressivenessLevel):
        """
        Given planning horizons, lateral and longitudinal polynomials and aggressiveness level, this method finds the
        optimal time w.r.t cost defined by aggressiveness-level-dependent weights.
        :param T_vals: np.ndarray of planning horizons among which the action time specification will be taken.
        :param poly_coefs_s: coefficients of longitudinal polynomial [np.ndarray]
        :param poly_lib_s: library of type [Poly1D] for kinematic calculations.
        :param poly_coefs_d: coefficients of lateral polynomial [np.ndarray]
        :param poly_lib_d: library of type [Poly1D] for kinematic calculations.
        :param agg_level: [AggressivenessLevel]
        :return: a tuple of (optimum time horizon, whether this time horizon meets acceleration and velocity constraints)
        """
        jerk_s = poly_lib_s.cumulative_jerk(poly_coefs_s, T_vals)
        jerk_d = poly_lib_d.cumulative_jerk(poly_coefs_d, T_vals)

        cost = np.dot(np.c_[jerk_s, jerk_d, T_vals],
                      np.c_[BP_JERK_S_JERK_D_TIME_WEIGHTS[agg_level.value]])
        optimum_time_idx = np.argmin(cost)

        are_lon_acc_in_limits = poly_lib_s.are_accelerations_in_limits(poly_coefs_s, T_vals, LON_ACC_LIMITS)
        are_lat_acc_in_limits = poly_lib_d.are_accelerations_in_limits(poly_coefs_d, T_vals, LAT_ACC_LIMITS)
        are_vel_in_limits = poly_lib_s.are_velocities_in_limits(poly_coefs_s, T_vals, VELOCITY_LIMITS)

        optimum_time_satisfies_constraints = are_lon_acc_in_limits[optimum_time_idx] and \
                                             are_lat_acc_in_limits[optimum_time_idx] and \
                                             are_vel_in_limits[optimum_time_idx]

        return optimum_time_idx, optimum_time_satisfies_constraints

    @staticmethod
    def define_lon_constraints(repeat_factor: int, ego_init_fstate: FrenetState2D, desired_acc: float,
                               desired_vel: np.ndarray, desired_lon: np.ndarray = None):
        """
        Defines longitudinal constraints for Werling trajectory planning
        :param repeat_factor: number of planning horizons, determines the shape of returned tensor.
        :param ego_init_fstate: ego initial frenet-frame state
        :param desired_acc: desired acceleration when action is finished (for each planning horizon)
        :param desired_vel: desired velocity when action is finished(for each planning horizon)
        :param desired_lon: desired longitudinal position when action is finished (for each planning horizon, optional)
        :return: a tensor with the constraints of third-order dynamics in initial and terminal action phase.
        """
        if desired_lon:
            # Quintic polynomial constraints
            constraints_s = np.c_[np.full(shape=repeat_factor, fill_value=ego_init_fstate[FS_SX]),
                                  np.full(shape=repeat_factor, fill_value=ego_init_fstate[FS_SV]),
                                  np.full(shape=repeat_factor, fill_value=ego_init_fstate[FS_SA]),
                                  desired_lon,
                                  desired_vel,
                                  np.full(shape=repeat_factor, fill_value=desired_acc)]
        else:
            # Quartic polynomial constraints (no constraint on sT)
            constraints_s = np.repeat([[
                ego_init_fstate[FS_SX],
                ego_init_fstate[FS_SV],
                ego_init_fstate[FS_SA],
                desired_vel,  # desired velocity
                0.0  # zero acceleration at the end of action
            ]], repeats=repeat_factor, axis=0)

        return constraints_s

    @staticmethod
    def define_lat_constraints(repeat_factor: int, ego_init_fstate: FrenetState2D, desired_lat: float):
        """
        Defines lateral constraints for Werling trajectory planning
        :param repeat_factor: number of planning horizons, determines the shape of returned tensor.
        :param ego_init_fstate: ego initial frenet-frame state
        :param desired_lat: desired lateral position when action is finished (for each planning horizon)
        :return: a tensor with the constraints of third-order dynamics in initial and terminal action phase.
        """
        # Quintic polynomial constraints
        constraints_d = np.repeat([[
            ego_init_fstate[FS_DX],
            ego_init_fstate[FS_DV],
            ego_init_fstate[FS_DA],
            desired_lat,
            0.0,
            0.0
        ]], repeats=repeat_factor, axis=0)

        return constraints_d


class StaticActionSpace(ActionSpace):

    def __init__(self, logger):
        super().__init__(logger)
        # TODO: should we get velocity_grid from constructor?
        self.velocity_grid = np.append(0,
                                       np.arange(MIN_VELOCITY, MAX_VELOCITY + np.finfo(np.float16).eps, VELOCITY_STEP))
        for comb in cartesian([RelativeLane, self.velocity_grid, AggressivenessLevel]):
            self._recipes.append(StaticActionRecipe(comb[0], comb[1], comb[2]))

    def _init_filters(self):
        self.recipe_filtering.add_filter(
            RecipeFilter(name='FilterIfNone', filtering_method=recipe_filter_methods.filter_if_none), is_active=True)
        self.recipe_filtering.add_filter(
            RecipeFilter(name='AlwaysFalse', filtering_method=recipe_filter_methods.always_false), is_active=True)

    def specify_goal(self, action_recipe: StaticActionRecipe, behavioral_state: SemanticBehavioralGridState) -> Optional[ActionSpec]:
        ego = behavioral_state.ego_state
        ego_init_cstate = np.array([ego.x, ego.y, ego.yaw, ego.v_x, ego.acceleration_lon, ego.curvature])
        road_id = ego.road_localization.road_id
        road_points = MapService.get_instance()._shift_road_points_to_latitude(road_id, 0.0)  # TODO: use nav_plan
        road_frenet = FrenetSerret2DFrame(road_points)
        ego_init_fstate = road_frenet.cstate_to_fstate(ego_init_cstate)

        road_lane_latitudes = MapService.get_instance().get_center_lanes_latitudes(road_id)
        desired_lane = ego.road_localization.lane_num + action_recipe.rel_lane.value
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
            self.logger.warning("Can\'t specify Recipe %s given ego state %s ", str(action_recipe), str(ego))
            return None

        # Note: We create the samplable trajectory as a reference trajectory of the current action.from
        # We assume correctness only of the longitudinal axis, and set T_d to be equal to T_s.
        samplable_trajectory = SamplableWerlingTrajectory(timestamp_in_sec=ego.timestamp_in_sec,
                                                          T_s=T_vals[optimum_time_idx],
                                                          T_d=T_vals[optimum_time_idx],
                                                          frenet_frame=road_frenet,
                                                          poly_s_coefs=poly_coefs_s[optimum_time_idx],
                                                          poly_d_coefs=poly_coefs_d[optimum_time_idx])

        target_s = Math.zip_polyval2d(poly_coefs_s, T_vals[optimum_time_idx])

        return ActionSpec(t=T_vals[optimum_time_idx], v=constraints_s[optimum_time_idx, 3],
                          s=target_s[optimum_time_idx, 0],
                          d=constraints_d[optimum_time_idx, 3],
                          samplable_trajectory=samplable_trajectory)


class DynamicActionSpace(ActionSpace):

    def __init__(self, logger: Logger, predictor: Predictor):
        super().__init__(logger)
        for comb in cartesian(
                [RelativeLane, RelativeLongitudinalPosition, [ActionType.FOLLOW_VEHICLE, ActionType.TAKE_OVER_VEHICLE],
                 AggressivenessLevel]):
            self._recipes.append(DynamicActionRecipe(comb[0], comb[1], comb[2], comb[3]))
        self.predictor = predictor

    def _init_filters(self):
        self.recipe_filtering.add_filter(
            RecipeFilter(name='AlwaysTrue', filtering_method=recipe_filter_methods.filter_if_none), is_active=True)
        self.recipe_filtering.add_filter(
            RecipeFilter(name='AlwaysFalse', filtering_method=recipe_filter_methods.always_false), is_active=False)
        self.recipe_filtering.add_filter(RecipeFilter(name="FilterActionsTowardsNonOccupiedCells",
                                                      filtering_method=recipe_filter_methods.filter_actions_towards_non_occupied_cells),
                                         is_active=True)
        self.recipe_filtering.add_filter(RecipeFilter(name="FilterActionsTowardBackAndParallelCells",
                                                      filtering_method=recipe_filter_methods.filter_actions_toward_back_and_parallel_cells),
                                         is_active=True)
        self.recipe_filtering.add_filter(RecipeFilter(name="FilterOverTakeActions",
                                                      filtering_method=recipe_filter_methods.filter_over_take_actions),
                                         is_active=True)

    def specify_goal(self, action_recipe: DynamicActionRecipe,
                     behavioral_state: SemanticBehavioralGridState) -> Optional[ActionSpec]:
        """
        Given a state and a high level SemanticAction towards an object, generate a SemanticActionSpec.
        Internally, the reference route here is the RHS of the road, and the ActionSpec is specified with respect to it.
        :param action_recipe:
        :param behavioral_state: Frenet state of ego at initial point
        :return: semantic action specification
        """
        ego = behavioral_state.ego_state
        ego_init_cstate = np.array([ego.x, ego.y, ego.yaw, ego.v_x, ego.acceleration_lon, ego.curvature])
        road_id = ego.road_localization.road_id
        road_points = MapService.get_instance()._shift_road_points_to_latitude(road_id, 0.0)  # TODO: use nav_plan
        road_frenet = FrenetSerret2DFrame(road_points)
        ego_init_fstate = road_frenet.cstate_to_fstate(ego_init_cstate)

        target_obj = behavioral_state.road_occupancy_grid[(action_recipe.rel_lane.value, action_recipe.rel_lon.value)][
            0]
        target_obj_fpoint = road_frenet.cpoint_to_fpoint(np.array([target_obj.x, target_obj.y]))
        _, _, _, road_curvature_at_obj_location, _ = road_frenet._taylor_interp(target_obj_fpoint[FP_SX])
        obj_init_fstate = road_frenet.cstate_to_fstate(np.array([
            target_obj.x, target_obj.y,
            target_obj.yaw,
            target_obj.total_speed,
            target_obj.acceleration_lon,
            road_curvature_at_obj_location  # We don't care about other agent's curvature, only the road's
        ]))

        # Extract relevant details from state on Reference-Object
        obj_on_road = target_obj.road_localization
        road_lane_latitudes = MapService.get_instance().get_center_lanes_latitudes(road_id=obj_on_road.road_id)
        obj_center_lane_latitude = road_lane_latitudes[obj_on_road.lane_num]

        T_vals = np.arange(BP_ACTION_T_LIMITS[LIMIT_MIN], BP_ACTION_T_LIMITS[LIMIT_MAX] + np.finfo(np.float16).eps,
                           BP_ACTION_T_RES)

        # TODO: should be swapped with current implementation of Predictor.predict_object_on_road
        obj_saT = 0  # obj_init_fstate[FS_SA]
        obj_svT = obj_init_fstate[FS_SV] + obj_saT * T_vals
        obj_sxT = obj_init_fstate[FS_SX] + obj_svT * T_vals + obj_saT * T_vals ** 2 / 2

        safe_lon_dist = obj_svT * SAFE_DIST_TIME_DELAY
        lon_margin = SemanticActionsUtils.get_ego_lon_margin(ego.size) + target_obj.size.length / 2

        if action_recipe.action_type == ActionType.FOLLOW_VEHICLE:
            desired_lon = obj_sxT - safe_lon_dist - lon_margin
        elif action_recipe.action_type == ActionType.TAKE_OVER_VEHICLE:
            desired_lon = obj_sxT + safe_lon_dist + lon_margin
        else:
            raise NotImplemented("Action Type %s is not handled in DynamicActionSpace specification", action_recipe.action_type)

        constraints_s = ActionSpace.define_lon_constraints(len(T_vals), ego_init_fstate, obj_saT, obj_svT, desired_lon)
        constraints_d = ActionSpace.define_lat_constraints(len(T_vals), ego_init_fstate, obj_center_lane_latitude)
        A_inv = np.linalg.inv(QuinticPoly1D.time_constraints_tensor(T_vals))

        # solve for s(t) and d(t)
        poly_coefs_s = QuinticPoly1D.zip_solve(A_inv, constraints_s)
        poly_coefs_d = QuinticPoly1D.zip_solve(A_inv, constraints_d)

        optimum_time_idx, optimum_time_satisfies_constraints = ActionSpace.find_optimum_planning_time(T_vals,
                                                                                                      poly_coefs_s,
                                                                                                      QuinticPoly1D,
                                                                                                      poly_coefs_d,
                                                                                                      QuinticPoly1D,
                                                                                                      action_recipe.aggressiveness)

        if not optimum_time_satisfies_constraints:
            self.logger.warning("Can\'t specify Recipe %s given ego state %s ", str(action_recipe), str(ego))
            return None

        # Note: We create the samplable trajectory as a reference trajectory of the current action.from
        # We assume correctness only of the longitudinal axis, and set T_d to be equal to T_s.
        samplable_trajectory = SamplableWerlingTrajectory(timestamp_in_sec=ego.timestamp_in_sec,
                                                          T_s=T_vals[optimum_time_idx],
                                                          T_d=T_vals[optimum_time_idx],
                                                          frenet_frame=road_frenet,
                                                          poly_s_coefs=poly_coefs_s[optimum_time_idx],
                                                          poly_d_coefs=poly_coefs_d[optimum_time_idx])

        return ActionSpec(t=T_vals[optimum_time_idx], v=obj_svT[optimum_time_idx],
                          s=constraints_s[optimum_time_idx, 3],
                          d=constraints_d[optimum_time_idx, 3],
                          samplable_trajectory=samplable_trajectory)


class CombinedActionSpace(ActionSpace):
    def __init__(self, logger: Logger, static_action_space: StaticActionSpace,
                 dynamic_action_space: DynamicActionSpace):
        super().__init__(logger)
        self.static_action_space = static_action_space
        self.dynamic_action_space = dynamic_action_space

    @property
    def action_space_size(self) -> int:
        return self.static_action_space.action_space_size + self.dynamic_action_space.action_space_size

    @property
    def recipes(self) -> List[ActionRecipe]:
        return self.static_action_space.recipes + self.dynamic_action_space.recipes

    @raises(NotImplemented)
    def specify_goal(self, action_recipe: ActionRecipe, behavioral_state: SemanticBehavioralGridState) -> ActionSpec:
        if isinstance(action_recipe, StaticActionRecipe):
            return self.static_action_space.specify_goal(action_recipe, behavioral_state)
        elif isinstance(action_recipe, DynamicActionRecipe):
            return self.dynamic_action_space.specify_goal(action_recipe, behavioral_state)
        else:
            raise NotImplemented('action_recipe %s is not a StaticActionRecipe nor a DynamicActionRecipe',
                                 action_recipe)

    @raises(NotImplemented)
    def filter_recipe(self, action_recipe: ActionRecipe, behavioral_state: SemanticBehavioralGridState) -> bool:
        if isinstance(action_recipe, StaticActionRecipe):
            return self.static_action_space.filter_recipe(action_recipe, behavioral_state)
        elif isinstance(action_recipe, DynamicActionRecipe):
            return self.dynamic_action_space.filter_recipe(action_recipe, behavioral_state)
        else:
            raise NotImplemented('action_recipe %s is not a StaticActionRecipe nor a DynamicActionRecipe',
                                 action_recipe)
