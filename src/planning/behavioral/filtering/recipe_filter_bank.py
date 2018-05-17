import os

from decision_making.src.global_constants import BP_JERK_S_JERK_D_TIME_WEIGHTS
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionRecipe, DynamicActionRecipe, \
    RelativeLongitudinalPosition, ActionType, RelativeLane, AggressivenessLevel
from decision_making.src.planning.behavioral.filtering.recipe_filtering import RecipeFilter
from decision_making.src.planning.utils.file_utils import BinaryReadWrite
from decision_making.src.planning.utils.math import Math

from decision_making.test.planning.utils.optimal_control.quintic_poly_formulas import v_0_grid, a_0_grid, s_T_grid, v_T_grid
from mapping.src.service.map_service import MapService

# DynamicActionRecipe Filters


class FilterActionsTowardsNonOccupiedCells(RecipeFilter):
    def filter(self, recipe: DynamicActionRecipe, behavioral_state: BehavioralGridState) -> bool:
        recipe_cell = (recipe.relative_lane, recipe.relative_lon)
        return recipe_cell in behavioral_state.road_occupancy_grid


class FilterBadExpectedTrajectory(RecipeFilter):
    def __init__(self, predicates_dir: str):
        for filename in os.listdir(predicates_dir):
            if not filename.endswith(".bin"):
                continue

            predicate_path = str(os.path.join(predicates_dir, filename))
            wT, wJ = [float(filename.split('.bin')[0].split('_')[3]),
                      float(filename.split('.bin')[0].split('_')[5])]
            predicate_shape = (
                int(v_0_grid.shape[0]), int(a_0_grid.shape[0]), int(s_T_grid.shape[0]), int(v_T_grid.shape[0]))

            self.predicates[(wT, wJ)] = BinaryReadWrite.load(file_path=predicate_path, shape=predicate_shape)

    def filter(self, recipe: DynamicActionRecipe, behavioral_state: BehavioralGridState) -> bool:
        if (
                recipe.action_type == ActionType.FOLLOW_VEHICLE and recipe.relative_lon == RelativeLongitudinalPosition.FRONT) \
                or (
                        recipe.action_type == ActionType.OVER_TAKE_VEHICLE and recipe.relative_lon == RelativeLongitudinalPosition.REAR):
            ego_state = behavioral_state.ego_state
            recipe_cell = (recipe.relative_lane, recipe.relative_lon)
            if recipe_cell in behavioral_state.road_occupancy_grid:
                v_0 = ego_state.v_x
                a_0 = ego_state.acceleration_lon
                relative_dynamic_object = behavioral_state.road_occupancy_grid[recipe_cell][0]
                dynamic_object = relative_dynamic_object.dynamic_object
                # TODO: the following is not accurate because it returns "same-lon" cars distance as 0
                s_T = relative_dynamic_object.distance
                v_T = dynamic_object.v_x
                wJ, _, wT = BP_JERK_S_JERK_D_TIME_WEIGHTS[recipe.aggressiveness.value]
                predicate = self.predicates[(wT, wJ)]
                if recipe.action_type == ActionType.FOLLOW_VEHICLE:
                    return predicate[Math.ind_on_uniform_axis(v_0, v_0_grid),
                                     Math.ind_on_uniform_axis(a_0, a_0_grid),
                                     Math.ind_on_uniform_axis(s_T, s_T_grid),
                                     Math.ind_on_uniform_axis(v_T, v_T_grid)] > 0
                else:  # OVER_TAKE_VEHICLE
                    return predicate[Math.ind_on_uniform_axis(v_0, v_0_grid),
                                     Math.ind_on_uniform_axis(a_0, a_0_grid),
                                     Math.ind_on_uniform_axis(s_T, s_T_grid),
                                     Math.ind_on_uniform_axis(v_T, v_T_grid)] > 0
            else:
                return False
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


