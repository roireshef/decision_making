from decision_making.src.global_constants import BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED
from typing import List

from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionRecipe, DynamicActionRecipe, \
    RelativeLongitudinalPosition, ActionType, RelativeLane, AggressivenessLevel, StaticActionRecipe
from decision_making.src.planning.behavioral.filtering.recipe_filtering import RecipeFilter
from decision_making.src.utils.map_utils import MapUtils
from decision_making.src.utils.print_utils import PrintUtils


class FilterActionsTowardsNonOccupiedCells(RecipeFilter):
    def filter(self, recipes: List[DynamicActionRecipe], behavioral_state: BehavioralGridState) -> List[bool]:
        return [(recipe.relative_lane, recipe.relative_lon) in behavioral_state.road_occupancy_grid
                if recipe is not None else False for recipe in recipes]


class FilterActionsTowardsOtherLanes(RecipeFilter):
    def filter(self, recipes: List[ActionRecipe], behavioral_state: BehavioralGridState) -> List[bool]:
        return [recipe.relative_lane == RelativeLane.SAME_LANE
                if recipe is not None else False for recipe in recipes]


class FilterActionsTowardBackCells(RecipeFilter):
    def filter(self, recipes: List[DynamicActionRecipe], behavioral_state: BehavioralGridState) -> List[bool]:
        return [recipe.relative_lon != RelativeLongitudinalPosition.REAR
                if recipe is not None else False for recipe in recipes]


class FilterActionsTowardBackAndParallelCells(RecipeFilter):
    def filter(self, recipes: List[DynamicActionRecipe], behavioral_state: BehavioralGridState) -> List[bool]:
        return [recipe.relative_lon == RelativeLongitudinalPosition.FRONT
                if recipe is not None else False for recipe in recipes]


class FilterOvertakeActions(RecipeFilter):
    def filter(self, recipes: List[DynamicActionRecipe], behavioral_state: BehavioralGridState) -> List[bool]:
        return [recipe.action_type != ActionType.OVERTAKE_VEHICLE
                if recipe is not None else False for recipe in recipes]


class FilterActionsTowardsCellsWithoutRoadSigns(RecipeFilter):
    def filter(self, recipes: List[DynamicActionRecipe], behavioral_state: BehavioralGridState) -> List[bool]:
            return [len(MapUtils.get_static_traffic_flow_controls_s(
                behavioral_state.extended_lane_frames[recipe.relative_lane])) > 0
                    if recipe is not None else False for recipe in recipes]


# General ActionRecipe Filters

class FilterIfNone(RecipeFilter):
    def filter(self, recipes: List[ActionRecipe], behavioral_state: BehavioralGridState) -> List[bool]:
        return [(recipe and behavioral_state) is not None for recipe in recipes]


class FilterNonCalmActions(RecipeFilter):
    def filter(self, recipes: List[ActionRecipe], behavioral_state: BehavioralGridState) -> List[bool]:
        return [recipe.aggressiveness == AggressivenessLevel.CALM
                if recipe is not None else False for recipe in recipes]


class FilterIfNoLane(RecipeFilter):
    def filter(self, recipes: List[ActionRecipe], behavioral_state: BehavioralGridState) -> List[bool]:
        lane_id = behavioral_state.ego_state.map_state.lane_id
        return [(recipe.relative_lane == RelativeLane.SAME_LANE or
                len(MapUtils.get_adjacent_lane_ids(lane_id, recipe.relative_lane)) > 0)
                if recipe is not None else False for recipe in recipes]


class FilterIfAggressive(RecipeFilter):
    def filter(self, recipes: List[ActionRecipe], behavioral_state: BehavioralGridState) -> List[bool]:
        return [recipe.aggressiveness != AggressivenessLevel.AGGRESSIVE
                if recipe is not None else False for recipe in recipes]


class FilterLaneChanging(RecipeFilter):
    def filter(self, recipes: List[ActionRecipe], behavioral_state: BehavioralGridState) -> List[bool]:
        return [recipe.relative_lane == RelativeLane.SAME_LANE
                if recipe is not None else False for recipe in recipes]


class FilterSpeedingOverDesiredVelocityStatic(RecipeFilter):
    """ This filter only compares the target lane speed with an absolute speed limit.
    Does NOT compare against the lane's speed limit, as it is not clear at this stage, which lane segments are relevant.
    This will be tested at the action spec filter FilterForLaneSpeedLimits
    """
    def filter(self, recipes: List[StaticActionRecipe], behavioral_state: BehavioralGridState) -> List[bool]:
        return [recipe.velocity <= BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED
                if recipe is not None else False for recipe in recipes]


class FilterSpeedingOverDesiredVelocityDynamic(RecipeFilter):
    """ This filter only compares the target vehicle's speed with an absolute speed limit.
    Does NOT compare against the lane's speed limit, as it is not clear at this stage, which lane segments are relevant.
    This will be tested at the action spec filter FilterForLaneSpeedLimits
    """
    def filter(self, recipes: List[DynamicActionRecipe], behavioral_state: BehavioralGridState) -> List[bool]:
        return [behavioral_state.road_occupancy_grid
                [(recipe.relative_lane, recipe.relative_lon)][0].dynamic_object.velocity
                <= BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED
                if recipe is not None else False for recipe in recipes]

