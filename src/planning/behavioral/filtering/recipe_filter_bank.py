from decision_making.src.global_constants import BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED, \
    MINIMUM_REQUIRED_TRAJECTORY_TIME_HORIZON
from typing import List

from decision_making.src.messages.scene_static_enums import RoadObjectType
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionRecipe, DynamicActionRecipe, \
    RelativeLongitudinalPosition, ActionType, RelativeLane, AggressivenessLevel, StaticActionRecipe, \
    RoadSignActionRecipe
from decision_making.src.planning.behavioral.filtering.recipe_filtering import RecipeFilter
from decision_making.src.planning.types import FS_SX, FS_SV, SIGN_S
from decision_making.src.utils.map_utils import MapUtils


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
        # currently looks only for stop signs and stop bars
        return [len(MapUtils.get_stop_bar_and_stop_sign(
            behavioral_state.extended_lane_frames[recipe.relative_lane])) > 0
                if ((recipe is not None) and (recipe.relative_lane in behavioral_state.extended_lane_frames))
                else False
                for recipe in recipes]


class FilterStopActionIfTooSoon(RecipeFilter):
    EXTRA_MARGIN = 10.0  # margin required for low speeds, to avoid reverting decision to stop after slowing down
    SPEED_THRESHOLDS = [2, 3, 10, 100]  # in [m/s]
    DECELERATION_THRESHOLDS = [0.3, 0.5, 1.2, 1.5]  # in [m/s^2]
    CLOSE_ENOUGH = 3.0  # define acceptable distance [m] between stop_bar and stop_sign to be considered as related

    @staticmethod
    def deceleration_for_speed(speed: float) -> float:
        """ defined by the system requirements """
        for idx, speed_threshold in enumerate(FilterStopActionIfTooSoon.SPEED_THRESHOLDS):
            if speed < speed_threshold:
                return FilterStopActionIfTooSoon.DECELERATION_THRESHOLDS[idx]

    @staticmethod
    def get_closest_stop_bar_distance(action: RoadSignActionRecipe, behavioral_state: BehavioralGridState) -> float:
        """
        Returns the s value of the closest Stop Bar or Stop sign.
        If both types exist, prefer stop bar, if close enough to stop sign.
        No existence checks necessary, as it was already tested by FilterActionsTowardsCellsWithoutRoadSigns
        :param action: the action recipe to be considered
        :param behavioral_state: BehavioralGridState in context
        :return: distance to closest stop bar
        """
        target_lane_frenet = behavioral_state.extended_lane_frames[action.relative_lane]  # the target GFF
        stop_bar_and_stop_signs = MapUtils.get_stop_bar_and_stop_sign(target_lane_frenet)
        stop_signs = MapUtils.filter_flow_control_by_type(stop_bar_and_stop_signs, [RoadObjectType.StopSign])
        stop_bars = MapUtils.filter_flow_control_by_type(stop_bar_and_stop_signs,
                                                         [RoadObjectType.StopBar_Left, RoadObjectType.StopBar_Right])
        # either stop BAR or SIGN are guaranteed to exist, but not necessarily both
        if len(stop_bars) == 0:
            # only stop SIGN exists
            closest_sign = stop_signs[0]
        elif len(stop_signs) == 0:
            # only stop BAR exists
            closest_sign = stop_bars[0]
        # Both stop SIGN and BAR exist
        elif stop_bars[0][FS_SX] < stop_signs[0][FS_SX] + FilterStopActionIfTooSoon.CLOSE_ENOUGH:
            # stop BAR close to SIGN or closer than stop SIGN, select it
            closest_sign = stop_bars[0]
        else:
            # stop SIGN is closer
            closest_sign = stop_signs[0]

        return closest_sign[SIGN_S] - behavioral_state.projected_ego_fstates[action.relative_lane][FS_SX]

    @staticmethod
    def is_time_to_stop(recipe: RoadSignActionRecipe, behavioral_state: BehavioralGridState) -> bool:
        ego_speed = behavioral_state.projected_ego_fstates[recipe.relative_lane][FS_SV]
        target_deceleration = FilterStopActionIfTooSoon.deceleration_for_speed(ego_speed)

        distance_to_sign = FilterStopActionIfTooSoon.get_closest_stop_bar_distance(recipe, behavioral_state)

        # calculate time to stop assuming fixed deceleration
        # use the simple equation of distance=acceleration*time^2/2 = acceleration*(ego_speed/acceleration)^2/2
        time_to_decelerate = ego_speed / target_deceleration
        # if time_to_decelerate > MINIMUM_REQUIRED_TRAJECTORY_TIME_HORIZON:
        #     time_to_decelerate += MINIMUM_REQUIRED_TRAJECTORY_TIME_HORIZON  # margin to keep so that we don't run into BeyondSpecStaticTrafficFlowControlFilter
        distance_to_stop = target_deceleration * (time_to_decelerate ** 2) / 2
        return distance_to_stop + FilterStopActionIfTooSoon.EXTRA_MARGIN > distance_to_sign

    def filter(self, recipes: List[RoadSignActionRecipe], behavioral_state: BehavioralGridState) -> List[bool]:
        # pick one stop action, all have same ego speed
        mask = [(FilterStopActionIfTooSoon.is_time_to_stop(recipe, behavioral_state)) if recipe is not None
                else False for recipe in recipes]
        return mask


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

