import os

from decision_making.paths import Paths
from decision_making.src.global_constants import BP_JERK_S_JERK_D_TIME_WEIGHTS, LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionRecipe, DynamicActionRecipe, \
    RelativeLongitudinalPosition, ActionType, RelativeLane, AggressivenessLevel
from decision_making.src.planning.behavioral.filtering.recipe_filtering import RecipeFilter
from decision_making.src.planning.utils.file_utils import BinaryReadWrite
from decision_making.src.planning.utils.math import Math
from decision_making.test.planning.utils.optimal_control.quintic_poly_formulas import v_0_grid, a_0_grid, s_T_grid, \
    v_T_grid


# DynamicActionRecipe Filters


class FilterActionsTowardsNonOccupiedCells(RecipeFilter):
    def filter(self, recipe: DynamicActionRecipe, behavioral_state: BehavioralGridState) -> bool:
        recipe_cell = (recipe.relative_lane, recipe.relative_lon)
        return recipe_cell in behavioral_state.road_occupancy_grid


class FilterBadExpectedTrajectory(RecipeFilter):
    def __init__(self, predicates_dir: str):
        self.predicates = {}
        directory = Paths.get_resource_absolute_path_filename(predicates_dir)
        for filename in os.listdir(directory):
            if filename.endswith(".bin"):
                predicate_path = Paths.get_resource_absolute_path_filename('%s/%s' % (predicates_dir, filename))
                action_type = filename.split('.bin')[0].split('_predicate')[0]
                wT, wJ = [float(filename.split('.bin')[0].split('_')[4]),
                          float(filename.split('.bin')[0].split('_')[6])]
                if action_type == 'follow_lane':
                    predicate_shape = (
                        int(v_0_grid.shape[0]), int(a_0_grid.shape[0]), int(v_T_grid.shape[0]))
                else:
                    predicate_shape = (
                        int(v_0_grid.shape[0]), int(a_0_grid.shape[0]), int(s_T_grid.shape[0]), int(v_T_grid.shape[0]))
                predicate = BinaryReadWrite.load(file_path=predicate_path, shape=predicate_shape)
                self.predicates[(action_type, wT, wJ)] = predicate

    def filter(self, recipe: ActionRecipe, behavioral_state: BehavioralGridState) -> bool:
        action_type = recipe.action_type
        ego_state = behavioral_state.ego_state
        v_0 = ego_state.v_x
        a_0 = ego_state.acceleration_lon
        wJ, _, wT = BP_JERK_S_JERK_D_TIME_WEIGHTS[recipe.aggressiveness.value]
        if (action_type == ActionType.FOLLOW_VEHICLE and recipe.relative_lon == RelativeLongitudinalPosition.FRONT) \
                or (
                action_type == ActionType.OVER_TAKE_VEHICLE and recipe.relative_lon == RelativeLongitudinalPosition.REAR):
            recipe_cell = (recipe.relative_lane, recipe.relative_lon)
            if recipe_cell in behavioral_state.road_occupancy_grid:

                relative_dynamic_object = behavioral_state.road_occupancy_grid[recipe_cell][0]
                dynamic_object = relative_dynamic_object.dynamic_object
                margin_sign = -1 if recipe.action_type == ActionType.FOLLOW_VEHICLE else +1
                # TODO: the following is not accurate because it returns "same-lon" cars distance as 0
                s_T = relative_dynamic_object.distance + margin_sign * (LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT +
                                                                        ego_state.size.length / 2 + dynamic_object.size.length / 2)
                v_T = dynamic_object.v_x
                wJ, _, wT = BP_JERK_S_JERK_D_TIME_WEIGHTS[recipe.aggressiveness.value]
                predicate = self.predicates[(action_type.name.lower(), wT, wJ)]

                return predicate[Math.ind_on_uniform_axis(v_0, v_0_grid),
                                 Math.ind_on_uniform_axis(a_0, a_0_grid),
                                 Math.ind_on_uniform_axis(s_T, margin_sign * s_T_grid),
                                 Math.ind_on_uniform_axis(v_T, v_T_grid)] > 0
            else:
                return False
        elif action_type == ActionType.FOLLOW_LANE:
            v_T = recipe.velocity
            predicate = self.predicates[(action_type.name.lower(), wT, wJ)]
            return predicate[Math.ind_on_uniform_axis(v_0, v_0_grid),
                             Math.ind_on_uniform_axis(a_0, a_0_grid),
                             Math.ind_on_uniform_axis(v_T, v_T_grid)] > 0
        else:
            return False


class FilterActionsTowardBackCells(RecipeFilter):
    def filter(self, recipe: DynamicActionRecipe, behavioral_state: BehavioralGridState):
        return recipe.relative_lon != RelativeLongitudinalPosition.REAR


class FilterActionsTowardBackAndParallelCells(RecipeFilter):
    def filter(self, recipe: DynamicActionRecipe, behavioral_state: BehavioralGridState) -> bool:
        return recipe.relative_lon == RelativeLongitudinalPosition.FRONT


class FilterOvertakeActions(RecipeFilter):
    def filter(self, recipe: DynamicActionRecipe, behavioral_state: BehavioralGridState) -> bool:
        return recipe.action_type != ActionType.OVER_TAKE_VEHICLE


# StaticActionRecipe Filters

class FilterIfNone(RecipeFilter):
    def filter(self, recipe: DynamicActionRecipe, behavioral_state: BehavioralGridState) -> bool:
        return (recipe and behavioral_state) is not None


class FilterNonCalmActions(RecipeFilter):
    def filter(self, recipe: ActionRecipe, behavioral_state: BehavioralGridState) -> bool:
        return recipe.aggressiveness == AggressivenessLevel.CALM


class FilterIfNoLane(RecipeFilter):
    def filter(self, recipe: ActionRecipe, behavioral_state: BehavioralGridState) -> bool:
        return (recipe.relative_lane == RelativeLane.SAME_LANE or
                (recipe.relative_lane == RelativeLane.RIGHT_LANE and behavioral_state.right_lane_exists) or
                (recipe.relative_lane == RelativeLane.LEFT_LANE and behavioral_state.left_lane_exists))
