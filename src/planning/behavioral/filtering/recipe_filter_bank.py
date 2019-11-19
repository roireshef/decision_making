from decision_making.src.global_constants import BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED, LANE_CHANGE_DELAY
from typing import List

from decision_making.src.messages.turn_signal_message import TurnSignalState
from decision_making.src.planning.behavioral.state.behavioral_grid_state import BehavioralGridState
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
        return [len(MapUtils.get_stop_bar_and_stop_sign(
            behavioral_state.extended_lane_frames[recipe.relative_lane])) > 0
                if ((recipe is not None) and (recipe.relative_lane in behavioral_state.extended_lane_frames))
                else False
                for recipe in recipes]


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
    def filter(self, recipes: List[ActionRecipe], behavioral_state: BehavioralGridState) -> List[bool]:
        # the if statement in the ternary operator is executed first and will short circuit if False,
        # so a KeyError will not happen when accessing the extended_lane_frames dict
        time_since_turn_signal_changed = behavioral_state.ego_state.timestamp_in_sec - \
                                         behavioral_state.ego_state.turn_signal.s_Data.s_time_changed.timestamp_in_seconds
        turn_signal_state = behavioral_state.ego_state.turn_signal.s_Data.e_e_turn_signal_state
        lane_change_active = behavioral_state.ego_state.lane_change_info.lane_change_active
        is_in_target_lane = behavioral_state.ego_state.lane_change_info.in_target_lane
        return [(recipe.relative_lane == RelativeLane.SAME_LANE
                 or behavioral_state.extended_lane_frames[recipe.relative_lane].gff_type in [GFFType.Augmented, GFFType.AugmentedPartial]
                 or ((recipe.relative_lane == RelativeLane.LEFT_LANE) and (turn_signal_state == TurnSignalState.CeSYS_e_LeftTurnSignalOn)
                     and (time_since_turn_signal_changed > LANE_CHANGE_DELAY)
                     and not (lane_change_active and is_in_target_lane))
                 or ((recipe.relative_lane == RelativeLane.RIGHT_LANE) and (turn_signal_state == TurnSignalState.CeSYS_e_RightTurnSignalOn)
                     and (time_since_turn_signal_changed > LANE_CHANGE_DELAY)
                     and not (lane_change_active and is_in_target_lane)))
                if (recipe is not None) and (recipe.relative_lane in behavioral_state.extended_lane_frames)
                else False for recipe in recipes]



class FilterNonLCActionsDuringLC(RecipeFilter):
    def filter(self, recipes: List[ActionRecipe], behavioral_state: BehavioralGridState) -> List[bool]:
        """
        If the car is in the middle of a lane change, maintain actions towards the target lane if the blinker is still active.
        """
        lane_change_target_rel_lane = behavioral_state.ego_state.lane_change_info.destination_relative_lane
        turn_signal_state = behavioral_state.ego_state.turn_signal.s_Data.e_e_turn_signal_state
        turn_signal_matching = (lane_change_target_rel_lane == RelativeLane.LEFT_LANE and turn_signal_state == TurnSignalState.CeSYS_e_LeftTurnSignalOn) \
                            or (lane_change_target_rel_lane == RelativeLane.RIGHT_LANE and turn_signal_state == TurnSignalState.CeSYS_e_RightTurnSignalOn) \
                            or (lane_change_target_rel_lane == RelativeLane.SAME_LANE and turn_signal_state == TurnSignalState.CeSYS_e_Off)

        # if lane change is not active or the lane change is toward an augmented lane,
        # this filter should not filter any actions
        return [(not behavioral_state.ego_state.lane_change_info.is_lane_change_active())
                or behavioral_state.extended_lane_frames[recipe.relative_lane].gff_type in [GFFType.Augmented, GFFType.AugmentedPartial]
                or (recipe.relative_lane == lane_change_target_rel_lane and turn_signal_matching)
                if (recipe is not None) and (recipe.relative_lane in behavioral_state.extended_lane_frames)
                else False for recipe in recipes]


# the if statement in the ternary operator is executed first and will short circuit if False,
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

