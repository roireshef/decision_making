import os

from decision_making.src.global_constants import BP_JERK_S_JERK_D_TIME_WEIGHTS
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionRecipe, DynamicActionRecipe, \
    RelativeLongitudinalPosition, ActionType, RelativeLane, AggressivenessLevel
from decision_making.src.planning.behavioral.filtering.recipe_filtering import RecipeFilter
from decision_making.src.planning.utils.file_utils import BinaryReadWrite
from decision_making.src.planning.utils.math import Math
from decision_making.src.planning.utils.optimal_control.quintic_poly_formulas import v_0_grid, a_0_grid, s_T_grid, \
    v_T_grid
from mapping.src.service.map_service import MapService

# DynamicActionRecipe Filters


class FilterActionsTowardsNonOccupiedCells(RecipeFilter):
    def filter(self, recipe: DynamicActionRecipe, behavioral_state: BehavioralGridState) -> bool:
        recipe_cell = (recipe.relative_lane.value, recipe.relative_lon.value)
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
        if recipe.action_type == ActionType.FOLLOW_VEHICLE:
            ego_state = behavioral_state.ego_state
            recipe_cell = (recipe.relative_lane.value, recipe.relative_lon.value)
            if recipe_cell in behavioral_state.road_occupancy_grid:
                v_0 = ego_state.v_x
                a_0 = ego_state.acceleration_lon
                dynamic_object = behavioral_state.road_occupancy_grid[recipe_cell][0]
                v_T = dynamic_object.v_x
                default_navigation_plan = MapService.get_instance().get_road_based_navigation_plan(
                    current_road_id=ego_state.road_localization.road_id)
                # TODO: the following is not accurate because it returns "same-lon" cars distance as 0
                sT_front, sT_rear = \
                    SemanticBehavioralGridState.calc_relative_distances(behavioral_state.ego_state,
                                                                        default_navigation_plan, dynamic_object)
                s_T = sT_front if recipe_cell[1] == RelativeLongitudinalPosition.FRONT else sT_rear

                wJ, _, wT = BP_JERK_S_JERK_D_TIME_WEIGHTS[AggressivenessLevel.value]
                predicate = self.predicates[(wT, wJ)]
                return predicate[Math.ind_on_uniform_axis(v_0, v_0_grid),
                                 Math.ind_on_uniform_axis(a_0, a_0_grid),
                                 Math.ind_on_uniform_axis(s_T, s_T_grid),
                                 Math.ind_on_uniform_axis(v_T, v_T_grid)]
            else:
                return False
        else:
            return True


class FilterActionsTowardBackCells(RecipeFilter):
    def filter(self, recipe: DynamicActionRecipe, behavioral_state: BehavioralGridState):
        return recipe.relative_lon != RelativeLongitudinalPosition.REAR


class FilterActionsTowardBackAndParallelCells(RecipeFilter):
    def filter(self, recipe: DynamicActionRecipe, behavioral_state: BehavioralGridState) -> bool:
        return recipe.relative_lon == RelativeLongitudinalPosition.FRONT


class FilterOvertakeActions(RecipeFilter):
    def filter(self, recipe: DynamicActionRecipe, behavioral_state: BehavioralGridState) -> bool:
        return recipe.action_type != ActionType.TAKE_OVER_VEHICLE


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


