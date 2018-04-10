from decision_making.src.planning.behavioral.architecture.data_objects import ActionRecipe, DynamicActionRecipe, \
    RelativeLongitudinalPosition, ActionType
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

# DynamicActionRecipe Filters


def filter_actions_towards_non_occupied_cells(recipe: DynamicActionRecipe, behavioral_state: SemanticBehavioralGridState) -> bool:
    recipe_cell = (recipe.rel_lane.value, recipe.rel_lon.value)
    cell_exists = recipe_cell in behavioral_state.road_occupancy_grid
    return len(behavioral_state.road_occupancy_grid[recipe_cell]) > 0 if cell_exists else False


def filter_actions_toward_back_cells(recipe: DynamicActionRecipe, behavioral_state: SemanticBehavioralGridState) -> bool:
    return recipe.rel_lon != RelativeLongitudinalPosition.REAR


def filter_actions_toward_back_and_parallel_cells(recipe: DynamicActionRecipe, behavioral_state: SemanticBehavioralGridState) -> bool:
    return recipe.rel_lon == RelativeLongitudinalPosition.FRONT


def filter_over_take_actions(recipe: DynamicActionRecipe, behavioral_state: SemanticBehavioralGridState) -> bool:
    return recipe.action_type != ActionType.TAKE_OVER_VEHICLE

# StaticActionRecipe Filters
