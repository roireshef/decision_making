from decision_making.src.planning.behavioral.architecture.components.filtering.recipe_filtering import RecipeFilter
from decision_making.src.planning.behavioral.architecture.data_objects import ActionRecipe, DynamicActionRecipe, \
    RelativeLongitudinalPosition, ActionType, RelativeLane
from decision_making.src.planning.behavioral.architecture.semantic_behavioral_grid_state import \
    SemanticBehavioralGridState
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState


# NOTE: All methods have to get as input ActionRecipe (or one of its children) and  BehavioralState (or one of its
#       children) even if they don't actually use them.

# These methods are used as filters and are used to initialize ActionRecipeFilter objects.
# ActionRecipe filtering methods (common to both dynamic and static recipes)


def filter_if_none(recipe: ActionRecipe, behavioral_state: BehavioralState) -> bool:
    if recipe and behavioral_state:
        return True
    else:
        return False


def always_false(recipe: ActionRecipe, behavioral_state: BehavioralState) -> bool:
    return False


def filter_if_no_lane(recipe: ActionRecipe, behavioral_state: SemanticBehavioralGridState) -> bool:
    return (recipe.relative_lane == RelativeLane.SAME_LANE or
            (recipe.relative_lane == RelativeLane.RIGHT_LANE and behavioral_state.right_lane_exists) or
            (recipe.relative_lane == RelativeLane.LEFT_LANE and behavioral_state.left_lane_exists))

# DynamicActionRecipe Filters


def filter_actions_towards_non_occupied_cells(recipe: DynamicActionRecipe,
                                              behavioral_state: SemanticBehavioralGridState) -> bool:
    recipe_cell = (recipe.relative_lane.value, recipe.relative_lon.value)
    return recipe_cell in behavioral_state.road_occupancy_grid


def filter_actions_toward_back_cells(recipe: DynamicActionRecipe, behavioral_state: SemanticBehavioralGridState) -> bool:
    return recipe.relative_lon != RelativeLongitudinalPosition.REAR


def filter_actions_toward_back_and_parallel_cells(recipe: DynamicActionRecipe,
                                                  behavioral_state: SemanticBehavioralGridState) -> bool:
    return recipe.relative_lon == RelativeLongitudinalPosition.FRONT


def filter_over_take_actions(recipe: DynamicActionRecipe, behavioral_state: SemanticBehavioralGridState) -> bool:
    return recipe.action_type != ActionType.TAKE_OVER_VEHICLE


# StaticActionRecipe Filters


# Filter list definition
dynamic_filters = [RecipeFilter(name='filter_if_none', filtering_method=filter_if_none),
                   RecipeFilter(name="filter_actions_towards_non_occupied_cells",
                                filtering_method=filter_actions_towards_non_occupied_cells),
                   RecipeFilter(name="filter_actions_toward_back_and_parallel_cells",
                                filtering_method=filter_actions_toward_back_and_parallel_cells),
                   RecipeFilter(name="filter_over_take_actions",
                                filtering_method=filter_over_take_actions)]

static_filters = [RecipeFilter(name='filter_if_none', filtering_method=filter_if_none),
                  RecipeFilter(name='filter_if_no_lane', filtering_method=filter_if_no_lane)]
