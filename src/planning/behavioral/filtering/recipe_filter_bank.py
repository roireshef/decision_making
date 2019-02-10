from decision_making.src.exceptions import ResourcesNotUpToDateException
from decision_making.src.global_constants import *
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionRecipe, DynamicActionRecipe, \
    RelativeLongitudinalPosition, ActionType, RelativeLane, AggressivenessLevel
from decision_making.src.planning.behavioral.filtering.recipe_filtering import RecipeFilter
from decision_making.src.planning.types import FS_SV, FS_SA
# DynamicActionRecipe Filters
from decision_making.src.utils.map_utils import MapUtils


class FilterActionsTowardsNonOccupiedCells(RecipeFilter):
    def filter(self, recipe: DynamicActionRecipe, behavioral_state: BehavioralGridState) -> bool:
        recipe_cell = (recipe.relative_lane, recipe.relative_lon)
        return recipe_cell in behavioral_state.road_occupancy_grid


class FilterActionsTowardsOtherLanes(RecipeFilter):
    def filter(self, recipe: ActionRecipe, behavioral_state: BehavioralGridState) -> bool:
        return recipe.relative_lane == RelativeLane.SAME_LANE


class FilterLimitsViolatingTrajectory(RecipeFilter):
    """
    This filter checks velocity, acceleration and time limits
    """
    def __init__(self, predicates_dir: str):
        if not self.validate_predicate_constants(predicates_dir):
            raise ResourcesNotUpToDateException('Predicates files were creates with other set of constants')
        self.predicates = self.read_predicates(predicates_dir, 'limits')

    def filter(self, recipe: ActionRecipe, behavioral_state: BehavioralGridState) -> bool:
        """
        This filter checks if recipe might cause a bad action specification, meaning velocity or acceleration are too
        aggressive, action time is too long. Filtering is based on querying a boolean predicate (LUT) created offline.
        :param recipe:
        :param behavioral_state:
        :return: True if recipe is valid, otherwise False
        """
        action_type = recipe.action_type

        # The predicates currently work for follow-front car,overtake-back car or follow-lane actions.
        if (action_type == ActionType.FOLLOW_VEHICLE and recipe.relative_lon == RelativeLongitudinalPosition.FRONT) \
                or (
                action_type == ActionType.OVERTAKE_VEHICLE and recipe.relative_lon == RelativeLongitudinalPosition.REAR):

            return RecipeFilter.filter_follow_vehicle_action(recipe, behavioral_state, self.predicates)

        elif action_type == ActionType.FOLLOW_LANE:

            ego_state = behavioral_state.ego_state
            v_0 = ego_state.map_state.lane_fstate[FS_SV]
            a_0 = ego_state.map_state.lane_fstate[FS_SA]
            wJ, _, wT = BP_JERK_S_JERK_D_TIME_WEIGHTS[recipe.aggressiveness.value]
            v_T = recipe.velocity

            predicate = self.predicates[(action_type.name.lower(), wT, wJ)]

            return predicate[FILTER_V_0_GRID.get_index(v_0), FILTER_A_0_GRID.get_index(a_0),
                             FILTER_V_T_GRID.get_index(v_T)] > 0
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
        return recipe.action_type != ActionType.OVERTAKE_VEHICLE


# StaticActionRecipe Filters

class FilterIfNone(RecipeFilter):
    def filter(self, recipe: DynamicActionRecipe, behavioral_state: BehavioralGridState) -> bool:
        return (recipe and behavioral_state) is not None


class FilterNonCalmActions(RecipeFilter):
    def filter(self, recipe: ActionRecipe, behavioral_state: BehavioralGridState) -> bool:
        return recipe.aggressiveness == AggressivenessLevel.CALM


class FilterIfNoLane(RecipeFilter):
    def filter(self, recipe: ActionRecipe, behavioral_state: BehavioralGridState) -> bool:
        lane_id = behavioral_state.ego_state.map_state.lane_id
        return (recipe.relative_lane == RelativeLane.SAME_LANE or
                len(MapUtils.get_adjacent_lane_ids(lane_id, recipe.relative_lane)) > 0)


class FilterIfAggressive(RecipeFilter):
    def filter(self, recipe: ActionRecipe, behavioral_state: BehavioralGridState) -> bool:
        return recipe.aggressiveness != AggressivenessLevel.AGGRESSIVE

class FilterLaneChanging(RecipeFilter):
    def filter(self, recipe: ActionRecipe, behavioral_state: BehavioralGridState) -> bool:
        return recipe.relative_lane == RelativeLane.SAME_LANE