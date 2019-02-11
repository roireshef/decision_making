import os

from decision_making.src.exceptions import ResourcesNotUpToDateException
from decision_making.src.global_constants import *
from decision_making.paths import Paths
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionRecipe, DynamicActionRecipe, \
    RelativeLongitudinalPosition, ActionType, RelativeLane, AggressivenessLevel
from decision_making.src.planning.behavioral.filtering.recipe_filtering import RecipeFilter
from decision_making.src.planning.types import FS_SV, FS_SA
from decision_making.src.planning.utils.file_utils import BinaryReadWrite, TextReadWrite

from decision_making.src.planning.utils.numpy_utils import UniformGrid
# DynamicActionRecipe Filters
from decision_making.src.utils.map_utils import MapUtils
from typing import List


class FilterActionsTowardsNonOccupiedCells(RecipeFilter):
    def filter(self, recipes: List[DynamicActionRecipe], behavioral_state: BehavioralGridState) -> List[bool]:
        return [(recipe.relative_lane, recipe.relative_lon) in behavioral_state.road_occupancy_grid
                if recipe is not None else False for recipe in recipes]


class FilterActionsTowardsOtherLanes(RecipeFilter):
    def filter(self, recipes: List[ActionRecipe], behavioral_state: BehavioralGridState) -> List[bool]:
        return [recipe.relative_lane == RelativeLane.SAME_LANE
                if recipe is not None else False for recipe in recipes]


class FilterLimitsViolatingTrajectory(RecipeFilter):
    """
    This filter checks velocity, acceleration and time limits
    """
    def __init__(self, predicates_dir: str):
        if not self.validate_predicate_constants(predicates_dir):
            raise ResourcesNotUpToDateException('Predicates files were creates with other set of constants')
        self.predicates = self.read_predicates(predicates_dir, 'limits')

    @staticmethod
    def read_predicates(predicates_dir, filter_name):
        """
        This method reads boolean maps from file into a dictionary mapping a tuple of (action_type,weights) to a binary LUT.
        :param predicates_dir: The directory holding all binary maps (.bin files)
        :param filter_name: either 'limits' or 'safety'
        :return: a dictionary mapping a tuple of (action_type,weights) to a binary LUT.
        """
        directory = Paths.get_resource_absolute_path_filename(predicates_dir)
        predicates = {}
        for filename in os.listdir(directory):
            if (filename.endswith(".bin") or filename.endswith(".npy")) and filter_name in filename:
                predicate_path = Paths.get_resource_absolute_path_filename('%s/%s' % (predicates_dir, filename))
                action_type = filename.split('.bin')[0].split('_' + filter_name)[0]
                wT, wJ = [float(filename.split('.bin')[0].split('_')[4]),
                          float(filename.split('.bin')[0].split('_')[6])]
                if action_type == 'follow_lane':
                    predicate_shape = (len(FILTER_V_0_GRID), len(FILTER_A_0_GRID), len(FILTER_V_T_GRID))
                else:
                    predicate_shape = (len(FILTER_V_0_GRID), len(FILTER_A_0_GRID), len(FILTER_S_T_GRID), len(FILTER_V_T_GRID))
                if filename.endswith(".npy"):
                    predicates[(action_type, wT, wJ)] = np.load(file=predicate_path)
                else:
                    predicates[(action_type, wT, wJ)] = BinaryReadWrite.load(file_path=predicate_path, shape=predicate_shape)
        return predicates

    # TODO: Move to test folder when agent is in steady state (global constants don't get changed)
    @staticmethod
    def validate_predicate_constants(predicates_dir):
        """
        This method checks if the predicates were created with the constants that are used right now
        :param predicates_dir: predicates directory under resources directory
        :return: True if constants are the same, False otherwise
        """
        # For this method to work, global_constants have to be imported
        metadata_path = Paths.get_resource_absolute_path_filename('%s/%s' % (predicates_dir, 'PredicatesMetaData.txt'))
        metadata_content = TextReadWrite.read(metadata_path)
        for line in metadata_content[1:5]:
            const_name = line.split()[0]
            grid_def = line.split('(', 1)[1].split(')')[0]
            grid_start, grid_end, grid_res = float(grid_def.split()[0].split(',')[0]), \
                                             float(grid_def.split()[2].split(',')[0]), \
                                             float(grid_def.split()[4].split(',')[0])
            file_grid = UniformGrid([grid_start, grid_end], grid_res)
            if not globals()[const_name] == file_grid:
                return False
        for line in metadata_content[5:]:
            const_name = line.split()[0]
            const_value = float(line.split()[2])
            if not globals()[const_name] == const_value:
                return False
        return True

    def filter(self, recipes: List[ActionRecipe], behavioral_state: BehavioralGridState) -> List[bool]:
        """
        This filter checks if recipe might cause a bad action specification, meaning velocity or acceleration are too
        aggressive, action time is too long. Filtering is based on querying a boolean predicate (LUT) created offline.
        :param recipes:
        :param behavioral_state:
        :return: True if recipe is valid, otherwise False
        """
        filter_result = [True for i in range(len(recipes))]

        for i, recipe in enumerate(recipes):

            if recipe is None:
                filter_result[i] = False
                continue

            action_type = recipe.action_type
            ego_state = behavioral_state.ego_state
            v_0 = ego_state.map_state.lane_fstate[FS_SV]
            a_0 = ego_state.map_state.lane_fstate[FS_SA]
            wJ, _, wT = BP_JERK_S_JERK_D_TIME_WEIGHTS[recipe.aggressiveness.value]

            # The predicates currently work for follow-front car,overtake-back car or follow-lane actions.
            if (action_type == ActionType.FOLLOW_VEHICLE and recipe.relative_lon == RelativeLongitudinalPosition.FRONT) \
                    or (
                    action_type == ActionType.OVERTAKE_VEHICLE and recipe.relative_lon == RelativeLongitudinalPosition.REAR):

                recipe_cell = (recipe.relative_lane, recipe.relative_lon)

                if recipe_cell not in behavioral_state.road_occupancy_grid:
                    filter_result[i] = False
                    continue

                # pull target vehicle
                relative_dynamic_object = behavioral_state.road_occupancy_grid[recipe_cell][0]
                dynamic_object = relative_dynamic_object.dynamic_object
                # safety distance is behind or ahead of target vehicle if we follow or overtake it, respectively.
                margin_sign = +1 if recipe.action_type == ActionType.FOLLOW_VEHICLE else -1
                # compute distance from target vehicle +/- safety margin
                s_T = relative_dynamic_object.longitudinal_distance - (LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT +
                                                          ego_state.size.length / 2 + dynamic_object.size.length / 2)
                v_T = dynamic_object.map_state.lane_fstate[FS_SV]

                predicate = self.predicates[(action_type.name.lower(), wT, wJ)]

                filter_result[i] = predicate[FILTER_V_0_GRID.get_index(v_0), FILTER_A_0_GRID.get_index(a_0),
                                    FILTER_S_T_GRID.get_index(margin_sign * s_T), FILTER_V_T_GRID.get_index(v_T)] > 0

            elif action_type == ActionType.FOLLOW_LANE:

                v_T = recipe.velocity

                predicate = self.predicates[(action_type.name.lower(), wT, wJ)]

                filter_result[i] = predicate[FILTER_V_0_GRID.get_index(v_0), FILTER_A_0_GRID.get_index(a_0),
                                             FILTER_V_T_GRID.get_index(v_T)] > 0
            else:
                filter_result[i] = False

        return filter_result


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
