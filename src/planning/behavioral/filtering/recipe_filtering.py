from abc import ABCMeta, abstractmethod
import traceback
from decision_making.paths import Paths
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.types import FS_SV, FS_SA
from decision_making.src.planning.utils.file_utils import TextReadWrite, BinaryReadWrite
from logging import Logger
from typing import List, Optional, Dict
from decision_making.src.global_constants import *

import six

from decision_making.src.planning.behavioral.behavioral_state import BehavioralState
from decision_making.src.planning.behavioral.data_objects import ActionRecipe, ActionType, RelativeLongitudinalPosition
import os


@six.add_metaclass(ABCMeta)
class RecipeFilter:
    """
    Base class for filter implementations that act on ActionRecipe and returns a boolean value that corresponds to
    whether the ActionRecipe satisfies the constraint in the filter. All filters have to get as input ActionRecipe
    (or one of its children) and  BehavioralState (or one of its children) even if they don't actually use them.
    """

    @abstractmethod
    def filter(self, recipes: List[ActionRecipe], behavioral_state: BehavioralState) -> List[bool]:
        """
        Filters an ActionRecipe based on the state of ego and nearby vehicles (BehavioralState).
        :param recipes: an object representing the semantic action to be considered
        :param behavioral_state: semantic behavioral state, containing the semantic grid
        :return: A boolean result, True if recipe is valid and false if filtered
        """
        pass

    def __str__(self):
        return self.__class__.__name__

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

    @staticmethod
    def filter_follow_vehicle_action(recipe: ActionRecipe, behavioral_state: BehavioralGridState, predicates_dict: Dict) -> bool:
        """
        This filter checks if recipe might cause a bad action specification, meaning velocity or acceleration are too
        aggressive, action time is too long or safety will be violated by entering non-safe zone while action is being
        taken. Filtering is based on querying a boolean predicate (LUT) created offline.
        :param recipe:
        :param behavioral_state:
        :param predicates_dict: dictionary from tuple (action_type, weights) to boolean matrix
        :return: True if recipe is valid, otherwise False
        """
        action_type = recipe.action_type
        ego_state = behavioral_state.ego_state
        v_0 = ego_state.map_state.lane_fstate[FS_SV]
        a_0 = ego_state.map_state.lane_fstate[FS_SA]
        wJ, _, wT = BP_JERK_S_JERK_D_TIME_WEIGHTS[recipe.aggressiveness.value]

        # The predicates currently work for follow-front car,overtake-back car or follow-lane actions.
        recipe_cell = (recipe.relative_lane, recipe.relative_lon)

        if recipe_cell not in behavioral_state.road_occupancy_grid:
            return False

        # pull target vehicle
        relative_dynamic_object = behavioral_state.road_occupancy_grid[recipe_cell][0]
        dynamic_object = relative_dynamic_object.dynamic_object
        # safety distance is behind or ahead of target vehicle if we follow or overtake it, respectively.
        margin_sign = +1 if recipe.action_type == ActionType.FOLLOW_VEHICLE else -1
        # compute distance from target vehicle +/- safety margin
        s_T = relative_dynamic_object.longitudinal_distance - (LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT +
                                                  ego_state.size.length / 2 + dynamic_object.size.length / 2)
        v_T = dynamic_object.map_state.lane_fstate[FS_SV]

        predicate = predicates_dict[(action_type.name.lower(), wT, wJ)]
        return predicate[FILTER_V_0_GRID.get_index(v_0), FILTER_A_0_GRID.get_index(a_0),
                         FILTER_S_T_GRID.get_index(margin_sign * s_T), FILTER_V_T_GRID.get_index(v_T)] > 0


class RecipeFiltering:
    """
    The gateway to execute filtering on one (or more) ActionRecipe(s). From efficiency point of view, the filters
    should be sorted from the strongest (the one filtering the largest number of recipes) to the weakest.
    """

    def __init__(self, filters: Optional[List[RecipeFilter]], logger: Logger):
        self._filters: List[RecipeFilter] = filters or []
        self.logger = logger

    def filter_recipes(self, recipes: List[ActionRecipe], behavioral_state: BehavioralState) -> List[bool]:
        """
        Filters a list of 'ActionRecipe's based on the state of ego and nearby vehicles (BehavioralState).
        :param recipes: A list of objects representing the semantic actions to be considered
        :param behavioral_state: semantic behavioral state, containing the semantic grid
        :return: A boolean List , True where the respective recipe is valid and false where it is filtered
        """
        mask = [True]*len(recipes)
        for recipe_filter in self._filters:
            mask = recipe_filter.filter(recipes, behavioral_state)
            recipes = [recipes[i] if mask[i] else None for i in range(len(recipes))]
        return mask

    def filter_recipe(self, recipe: ActionRecipe, behavioral_state: BehavioralState) -> bool:
        """
        Filters an 'ActionRecipe's based on the state of ego and nearby vehicles (BehavioralState).
        :param recipe: An object representing the semantic actions to be considered
        :param behavioral_state: semantic behavioral state, containing the semantic grid
        :return: A boolean , True where the recipe is valid and false where it is filtered
        """
        return self.filter_recipes([recipe], behavioral_state)[0]
