from decision_making.src.global_constants import BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED, LANE_CHANGE_DELAY
from typing import List

from decision_making.src.planning.behavioral.state.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.state.lane_change_state import LaneChangeStatus
from decision_making.src.planning.behavioral.data_objects import ActionRecipe, DynamicActionRecipe, \
    RelativeLongitudinalPosition, ActionType, RelativeLane, AggressivenessLevel, StaticActionRecipe, \
    RoadSignActionRecipe
from decision_making.src.planning.behavioral.filtering.recipe_filtering import RecipeFilter
from decision_making.src.utils.map_utils import MapUtils
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GFFType


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


class FilterActionsTowardsCellsWithoutStopSignsOrStopBars(RecipeFilter):
    def filter(self, recipes: List[RoadSignActionRecipe], behavioral_state: BehavioralGridState) -> List[bool]:
        return [behavioral_state.get_closest_stop_bar(recipe.relative_lane) is not None
                if ((recipe is not None) and (recipe.relative_lane in behavioral_state.extended_lane_frames))
                else False for recipe in recipes]


class FilterRoadSignActions(RecipeFilter):
    """ The purpose of this filter is to temporarily disable the stop for geo location feature """
    def filter(self, recipes: List[RoadSignActionRecipe], behavioral_state: BehavioralGridState) -> List[bool]:
        return [recipe.action_type != ActionType.FOLLOW_ROAD_SIGN
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
        return [(recipe.relative_lane == RelativeLane.SAME_LANE
                 or len(MapUtils.get_adjacent_lane_ids(lane_id, recipe.relative_lane)) > 0
                 or behavioral_state.extended_lane_frames[recipe.relative_lane].gff_type in [GFFType.Augmented, GFFType.AugmentedPartial])
                if (recipe is not None) and (recipe.relative_lane in behavioral_state.extended_lane_frames)
                else False for recipe in recipes]


class FilterIfAggressive(RecipeFilter):
    def filter(self, recipes: List[ActionRecipe], behavioral_state: BehavioralGridState) -> List[bool]:
        return [recipe.aggressiveness != AggressivenessLevel.AGGRESSIVE
                if recipe is not None else False for recipe in recipes]


class FilterLaneChangingIfNotAugmented(RecipeFilter):
    def filter(self, recipes: List[ActionRecipe], behavioral_state: BehavioralGridState) -> List[bool]:
        # the if statement in the ternary operator is executed first and will short circuit if False,
        # so a KeyError will not happen when accessing the extended_lane_frames dict
        return [(recipe.relative_lane == RelativeLane.SAME_LANE
                 or behavioral_state.extended_lane_frames[recipe.relative_lane].gff_type in [GFFType.Augmented, GFFType.AugmentedPartial])
                if (recipe is not None) and (recipe.relative_lane in behavioral_state.extended_lane_frames)
                else False for recipe in recipes]


class FilterLaneChangingIfNotAugmentedOrLaneChangeDesired(RecipeFilter):
    """
    This filter denies actions towards the LEFT or RIGHT lanes unless the lane is an augmented lane or a lane change is desired
    """
    def filter(self, recipes: List[ActionRecipe], behavioral_state: BehavioralGridState) -> List[bool]:
        lane_change_desired = behavioral_state.lane_change_state.status in \
                              [LaneChangeStatus.AnalyzingSafety, LaneChangeStatus.LaneChangeActiveInSourceLane]

        return [recipe.relative_lane == RelativeLane.SAME_LANE
                or behavioral_state.extended_lane_frames[recipe.relative_lane].gff_type in [GFFType.Augmented, GFFType.AugmentedPartial]
                or (lane_change_desired and recipe.relative_lane == behavioral_state.lane_change_state.target_relative_lane)
                if (recipe is not None) and (recipe.relative_lane in behavioral_state.extended_lane_frames)
                else False for recipe in recipes]


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

