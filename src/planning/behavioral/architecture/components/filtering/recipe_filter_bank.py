from decision_making.src.planning.behavioral.architecture.components.filtering.recipe_filtering import RecipeFilter
from decision_making.src.planning.behavioral.architecture.data_objects import ActionRecipe, DynamicActionRecipe, \
    RelativeLongitudinalPosition, ActionType, RelativeLane, AggressivenessLevel
from decision_making.src.planning.behavioral.architecture.semantic_behavioral_grid_state import \
    SemanticBehavioralGridState
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState


# NOTE: All methods have to get as input ActionRecipe (or one of its children) and  BehavioralState (or one of its
#       children) even if they don't actually use them.

# These methods are used as filters and are used to initialize ActionRecipeFilter objects.
# ActionRecipe filtering methods (common to both dynamic and static recipes)
from decision_making.src.planning.utils.math import Math
from mapping.src.service.map_service import MapService


def filter_if_none(recipe: ActionRecipe, behavioral_state: BehavioralState) -> bool:
    return (recipe and behavioral_state) is not None


def always_false(recipe: ActionRecipe, behavioral_state: BehavioralState) -> bool:
    return False


def filter_non_calm_actions(recipe: ActionRecipe, behavioral_state: SemanticBehavioralGridState) -> bool:
    return recipe.aggressiveness == AggressivenessLevel.CALM

# DynamicActionRecipe Filters


def filter_actions_towards_non_occupied_cells(recipe: DynamicActionRecipe,
                                              behavioral_state: SemanticBehavioralGridState) -> bool:
    recipe_cell = (recipe.relative_lane.value, recipe.relative_lon.value)
    return recipe_cell in behavioral_state.road_occupancy_grid


def filter_if_bad_expected_trajectory(recipe: DynamicActionRecipe,
                                      behavioral_state: SemanticBehavioralGridState) -> bool:
    ego_state = behavioral_state.ego_state
    if recipe.action_type == ActionType.FOLLOW_VEHICLE:
        recipe_cell = (recipe.relative_lane.value, recipe.relative_lon.value)
        if recipe_cell in behavioral_state.road_occupancy_grid:
            v_0 = ego_state.v_x
            a_0 = ego_state.acceleration_lon
            dynamic_object = behavioral_state.road_occupancy_grid[recipe_cell][0]
            v_T = dynamic_object.v_x
            default_navigation_plan = MapService.get_instance().get_road_based_navigation_plan(
                current_road_id=ego_state.road_localization.road_id)
            #TODO: the following is not accurate because it returns "same-lon" cars distance as 0
            sT_front, sT_rear = \
                SemanticBehavioralGridState.calc_relative_distances(behavioral_state.ego_state,
                                                                    default_navigation_plan, dynamic_object)
            s_T = sT_front if recipe_cell[1] == RelativeLongitudinalPosition.FRONT else sT_rear

            return predicate[Math.ind_on_uniform_axis(v_0, axis),
                             Math.ind_on_uniform_axis(a_0, axis),
                             Math.ind_on_uniform_axis(s_T, axis),
                             Math.ind_on_uniform_axis(v_T, axis)]
        else:
            return False




def filter_actions_toward_back_cells(recipe: DynamicActionRecipe, behavioral_state: SemanticBehavioralGridState) -> bool:
    return recipe.relative_lon != RelativeLongitudinalPosition.REAR


def filter_actions_toward_back_and_parallel_cells(recipe: DynamicActionRecipe,
                                                  behavioral_state: SemanticBehavioralGridState) -> bool:
    return recipe.relative_lon == RelativeLongitudinalPosition.FRONT


def filter_over_take_actions(recipe: DynamicActionRecipe, behavioral_state: SemanticBehavioralGridState) -> bool:
    return recipe.action_type != ActionType.TAKE_OVER_VEHICLE

# StaticActionRecipe Filters


def filter_if_no_lane(recipe: ActionRecipe, behavioral_state: SemanticBehavioralGridState) -> bool:
    return (recipe.relative_lane == RelativeLane.SAME_LANE or
            (recipe.relative_lane == RelativeLane.RIGHT_LANE and behavioral_state.right_lane_exists) or
            (recipe.relative_lane == RelativeLane.LEFT_LANE and behavioral_state.left_lane_exists))

# Note: From efficiency point of view, the filters should be sorted from the strongest (the one filtering the largest
# number of recipes) to the weakest.

# Filter list definition
dynamic_filters = [RecipeFilter(name='filter_if_none', filtering_method=filter_if_none),
                   RecipeFilter(name="filter_actions_towards_non_occupied_cells",
                                filtering_method=filter_actions_towards_non_occupied_cells),
                   RecipeFilter(name="filter_actions_toward_back_and_parallel_cells",
                                filtering_method=filter_actions_toward_back_and_parallel_cells),
                   RecipeFilter(name="filter_over_take_actions",
                                filtering_method=filter_over_take_actions),
                   RecipeFilter(name='filter_non_calm_actions', filtering_method=filter_non_calm_actions)]

static_filters = [RecipeFilter(name='filter_if_none', filtering_method=filter_if_none),
                  RecipeFilter(name='filter_if_no_lane', filtering_method=filter_if_no_lane),
                  RecipeFilter(name='filter_non_calm_actions', filtering_method=filter_non_calm_actions)]
