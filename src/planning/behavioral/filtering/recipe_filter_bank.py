import os

from decision_making.src.exceptions import ResourcesNotUpdatedException
from decision_making.src.global_constants import *
from decision_making.paths import Paths
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionRecipe, DynamicActionRecipe, \
    RelativeLongitudinalPosition, ActionType, RelativeLane, AggressivenessLevel
from decision_making.src.planning.behavioral.filtering.recipe_filtering import RecipeFilter
from decision_making.src.planning.utils.file_utils import BinaryReadWrite, TextReadWrite


# DynamicActionRecipe Filters
from decision_making.src.planning.utils.numpy_utils import UniformGrid


class FilterActionsTowardsNonOccupiedCells(RecipeFilter):
    def filter(self, recipe: DynamicActionRecipe, behavioral_state: BehavioralGridState) -> bool:
        recipe_cell = (recipe.relative_lane, recipe.relative_lon)
        return recipe_cell in behavioral_state.road_occupancy_grid


class FilterBadExpectedTrajectory(RecipeFilter):
    def __init__(self, predicates_dir: str):
        if not self.validate_predicate_constants(predicates_dir):
            raise ResourcesNotUpdatedException('Predicates files were creates with other set of constants')
        self.predicates = self.read_predicates(predicates_dir)

    @staticmethod
    def read_predicates(predicates_dir):
        directory = Paths.get_resource_absolute_path_filename(predicates_dir)
        predicates = {}
        for filename in os.listdir(directory):
            if filename.endswith(".bin"):
                predicate_path = Paths.get_resource_absolute_path_filename('%s/%s' % (predicates_dir, filename))
                action_type = filename.split('.bin')[0].split('_predicate')[0]
                wT, wJ = [float(filename.split('.bin')[0].split('_')[4]),
                          float(filename.split('.bin')[0].split('_')[6])]
                if action_type == 'follow_lane':
                    predicate_shape = (len(FILTER_V_0_GRID), len(FILTER_A_0_GRID), len(FILTER_V_T_GRID))
                else:
                    predicate_shape = (len(FILTER_V_0_GRID), len(FILTER_A_0_GRID), len(FILTER_S_T_GRID), len(FILTER_V_T_GRID))
                predicate = BinaryReadWrite.load(file_path=predicate_path, shape=predicate_shape)
                predicates[(action_type, wT, wJ)] = predicate

        return predicates

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

    def filter(self, recipe: ActionRecipe, behavioral_state: BehavioralGridState) -> bool:
        action_type = recipe.action_type
        ego_state = behavioral_state.ego_state
        v_0 = ego_state.v_x
        a_0 = ego_state.acceleration_lon
        wJ, _, wT = BP_JERK_S_JERK_D_TIME_WEIGHTS[recipe.aggressiveness.value]
        if (action_type == ActionType.FOLLOW_VEHICLE and recipe.relative_lon == RelativeLongitudinalPosition.FRONT) \
                or (
                action_type == ActionType.OVERTAKE_VEHICLE and recipe.relative_lon == RelativeLongitudinalPosition.REAR):
            recipe_cell = (recipe.relative_lane, recipe.relative_lon)
            if recipe_cell in behavioral_state.road_occupancy_grid:

                relative_dynamic_object = behavioral_state.road_occupancy_grid[recipe_cell][0]
                dynamic_object = relative_dynamic_object.dynamic_object
                margin_sign = +1 if recipe.action_type == ActionType.FOLLOW_VEHICLE else -1
                # TODO: the following is not accurate because it returns "same-lon" cars distance as 0
                s_T = relative_dynamic_object.distance -(LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT +
                                                                        ego_state.size.length / 2 + dynamic_object.size.length / 2)
                v_T = dynamic_object.v_x

                predicate = self.predicates[(action_type.name.lower(), wT, wJ)]

                # TODO: bug when s_T < 0 (on follow of near car)
                return predicate[FILTER_V_0_GRID.get_index(v_0), FILTER_A_0_GRID.get_index(a_0),
                                 FILTER_S_T_GRID.get_index(margin_sign*s_T), FILTER_V_T_GRID.get_index(v_T)] > 0
            else:
                return False
        elif action_type == ActionType.FOLLOW_LANE:
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
        return (recipe.relative_lane == RelativeLane.SAME_LANE or
                (recipe.relative_lane == RelativeLane.RIGHT_LANE and behavioral_state.right_lane_exists) or
                (recipe.relative_lane == RelativeLane.LEFT_LANE and behavioral_state.left_lane_exists))
