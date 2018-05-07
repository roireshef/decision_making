import os

from decision_making.paths import Paths
from decision_making.src.global_constants import BP_JERK_S_JERK_D_TIME_WEIGHTS
from decision_making.src.planning.behavioral.filtering.recipe_filtering import RecipeFilter
from decision_making.src.planning.behavioral.data_objects import ActionRecipe, DynamicActionRecipe, \
    RelativeLongitudinalPosition, ActionType, RelativeLane, AggressivenessLevel
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState
from decision_making.src.planning.utils.file_utils import BinaryReadWrite
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.planning.utils.math import Math
from decision_making.test.planning.utils.optimal_control.quintic_poly_formulas import v_0_grid, a_0_grid, s_T_grid, \
    v_T_grid
from mapping.src.service.map_service import MapService


# TODO: This code should be moved to some global system_init area
predicates = {}
directory = Paths.get_resource_absolute_path_filename('predicates/')
for filename in os.listdir(directory):
    if filename.endswith(".bin"):
        predicate_path = Paths.get_resource_absolute_path_filename('predicates/%s' % filename)
        wT, wJ = [float(filename.split('.bin')[0].split('_')[3]),
                  float(filename.split('.bin')[0].split('_')[5])]
        predicate_shape = (
            int(v_0_grid.shape[0]), int(a_0_grid.shape[0]), int(s_T_grid.shape[0]), int(v_T_grid.shape[0]))
        predicate = BinaryReadWrite.load(file_path=predicate_path, shape=predicate_shape)
        predicates[(wT, wJ)] = predicate


# NOTE: All methods have to get as input ActionRecipe (or one of its children) and  BehavioralState (or one of its
#       children) even if they don't actually use them.

# These methods are used as filters and are used to initialize ActionRecipeFilter objects.
# ActionRecipe filtering methods (common to both dynamic and static recipes)


def filter_if_none(recipe: ActionRecipe, behavioral_state: BehavioralState) -> bool:
    return (recipe and behavioral_state) is not None


def always_false(recipe: ActionRecipe, behavioral_state: BehavioralState) -> bool:
    return False


def filter_non_calm_actions(recipe: ActionRecipe, behavioral_state: BehavioralGridState) -> bool:
    return recipe.aggressiveness == AggressivenessLevel.CALM


# DynamicActionRecipe Filters


def filter_actions_towards_non_occupied_cells(recipe: DynamicActionRecipe,
                                              behavioral_state: BehavioralGridState) -> bool:
    recipe_cell = (recipe.relative_lane.value, recipe.relative_lon.value)
    return recipe_cell in behavioral_state.road_occupancy_grid


def filter_bad_expected_trajectory(recipe: DynamicActionRecipe,
                                   behavioral_state: BehavioralGridState) -> bool:
    if recipe.action_type == ActionType.FOLLOW_VEHICLE:
        ego_state = behavioral_state.ego_state
        recipe_cell = (recipe.relative_lane.value, recipe.relative_lon.value)
        if recipe_cell in behavioral_state.road_occupancy_grid:
            v_0 = ego_state.v_x
            a_0 = ego_state.acceleration_lon
            relative_dynamic_object = behavioral_state.road_occupancy_grid[recipe_cell][0]
            dynamic_object = relative_dynamic_object.dynamic_object
            # TODO: the following is not accurate because it returns "same-lon" cars distance as 0
            s_T = relative_dynamic_object.distance
            v_T = dynamic_object.v_x
            wJ,_,wT = BP_JERK_S_JERK_D_TIME_WEIGHTS[AggressivenessLevel.value]
            predicate = predicates[(wT, wJ)]
            return predicate[Math.ind_on_uniform_axis(v_0, v_0_grid),
                             Math.ind_on_uniform_axis(a_0, a_0_grid),
                             Math.ind_on_uniform_axis(s_T, s_T_grid),
                             Math.ind_on_uniform_axis(v_T, v_T_grid)]
        else:
            return False
    else:
        return True


def filter_actions_toward_back_cells(recipe: DynamicActionRecipe,
                                     behavioral_state: BehavioralGridState) -> bool:
    return recipe.relative_lon != RelativeLongitudinalPosition.REAR


def filter_actions_toward_back_and_parallel_cells(recipe: DynamicActionRecipe,
                                                  behavioral_state: BehavioralGridState) -> bool:
    return recipe.relative_lon == RelativeLongitudinalPosition.FRONT


def filter_over_take_actions(recipe: DynamicActionRecipe, behavioral_state: BehavioralGridState) -> bool:
    return recipe.action_type != ActionType.TAKE_OVER_VEHICLE


# StaticActionRecipe Filters


def filter_if_no_lane(recipe: ActionRecipe, behavioral_state: BehavioralGridState) -> bool:
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
                   RecipeFilter(name='filter_non_calm_actions', filtering_method=filter_non_calm_actions),
                   RecipeFilter(name='filter_bad_expected_trajectory', filtering_method=filter_bad_expected_trajectory)]

static_filters = [RecipeFilter(name='filter_if_none', filtering_method=filter_if_none),
                  RecipeFilter(name='filter_if_no_lane', filtering_method=filter_if_no_lane),
                  RecipeFilter(name='filter_non_calm_actions', filtering_method=filter_non_calm_actions)]
