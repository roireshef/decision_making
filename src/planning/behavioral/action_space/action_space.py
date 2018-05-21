import itertools
from abc import abstractmethod
from collections import defaultdict
from logging import Logger
from typing import List, Optional, Type

import numpy as np

from decision_making.src.exceptions import raises
from decision_making.src.global_constants import LON_ACC_LIMITS, LAT_ACC_LIMITS, VELOCITY_LIMITS, \
    BP_JERK_S_JERK_D_TIME_WEIGHTS
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState
from decision_making.src.planning.behavioral.data_objects import ActionSpec
from decision_making.src.planning.behavioral.data_objects import AggressivenessLevel, \
    ActionRecipe
from decision_making.src.planning.behavioral.filtering.recipe_filtering import RecipeFiltering
from decision_making.src.planning.types import FS_SV, FS_SX, FS_SA, FS_DX, FS_DV, FS_DA, \
    FrenetState2D, Limits, LIMIT_MIN, LIMIT_MAX
from decision_making.src.planning.utils.math import Math
from decision_making.src.planning.utils.optimal_control.poly1d import Poly1D


class ActionSpace:
    def __init__(self, logger: Logger, recipes: List[ActionRecipe], recipe_filtering: Optional[RecipeFiltering] = None):
        """
        Abstract class for Action-Space implementations. Implementations should include actions enumeration, filtering
         and specification.
        :param logger: dedicated logger implementation
        :param recipes: list of recipes that define the scope of an ActionSpace implementation
        :param recipe_filtering: RecipeFiltering object that holds the logic for filtering recipes
        """
        self.logger = logger
        self._recipes = recipes
        self._recipe_filtering = recipe_filtering or RecipeFiltering()

    @property
    def action_space_size(self) -> int:
        return len(self._recipes)

    @property
    def recipes(self) -> List[ActionRecipe]:
        return self._recipes

    @property
    @abstractmethod
    def recipe_classes(self) -> List[Type]:
        pass

    def filter_recipe(self, recipe: ActionRecipe, behavioral_state: BehavioralState) -> bool:
        return self._recipe_filtering.filter_recipe(recipe, behavioral_state)

    def filter_recipes(self, action_recipes: List[ActionRecipe], behavioral_state: BehavioralState):
        return self._recipe_filtering.filter_recipes(action_recipes, behavioral_state)

    @abstractmethod
    def specify_goal(self, action_recipe: ActionRecipe, behavioral_state: BehavioralGridState) -> Optional[ActionSpec]:
        """
        This method's purpose is to specify the enumerated actions that the agent can take.
        Each semantic action (action_recipe) is translated to a trajectory of the agent.
        The trajectory specification is created towards a target object in given cell in case of Dynamic action,
        and towards a certain lane in case of Static action, considering ego state.
        Internally, the reference route here is the RHS of the road, and the ActionSpec is specified with respect to it.
        :param action_recipe: an enumerated semantic action [ActionRecipe].
        :param behavioral_state: Frenet state of ego at initial point
        :return: semantic action specification [ActionSpec] or [None] if recipe can't be specified.
        """
        return self.specify_goals([action_recipe], behavioral_state)[0]

    @abstractmethod
    def specify_goals(self, action_recipes: List[ActionRecipe], behavioral_state: BehavioralGridState) -> List[Optional[ActionSpec]]:
        pass


class ActionSpaceContainer(ActionSpace):
    def __init__(self, logger: Logger, action_spaces: List[ActionSpace]):
        super().__init__(logger, [])
        self._action_spaces = action_spaces

        self._recipe_handler = {recipe_class: aspace
                                for aspace in action_spaces
                                for recipe_class in aspace.recipe_classes}

    @property
    def action_space_size(self) -> int:
        return sum(aspace.action_space_size for aspace in self._action_spaces)

    @property
    def recipes(self) -> List[ActionRecipe]:
        return list(itertools.chain.from_iterable(aspace.recipes for aspace in self._action_spaces))

    @property
    def recipe_classes(self) -> List:
        return list(itertools.chain.from_iterable(aspace.recipe_classes for aspace in self._action_spaces))

    def specify_goal(self, action_recipe: ActionRecipe, behavioral_state: BehavioralGridState) -> Optional[ActionSpec]:
        return self._recipe_handler[action_recipe.__class__].specify_goal(action_recipe, behavioral_state)

    def specify_goals(self, action_recipes: List[ActionRecipe], behavioral_state: BehavioralGridState) -> \
            List[Optional[ActionSpec]]:
        grouped_actions = defaultdict(list)
        grouped_idxs = defaultdict(list)
        for idx, recipe in enumerate(action_recipes):
            grouped_actions[recipe.__class__].append(recipe)
            grouped_idxs[recipe.__class__].append(idx)

        indexed_action_specs = list(itertools.chain.from_iterable(
            [zip(grouped_idxs[action_class], self._recipe_handler[action_class].specify_goals(action_list, behavioral_state))
             for action_class, action_list in grouped_actions.items()]))

        return [action for idx, action in sorted(indexed_action_specs, key=lambda idx_action: idx_action[0])]

    def filter_recipe(self, action_recipe: ActionRecipe, behavioral_state: BehavioralState) -> bool:
        return self._recipe_handler[action_recipe.__class__].filter_recipe(action_recipe, behavioral_state)

    # TODO: figure out how to remove the for loop for better efficiency and stay consistent with ordering
    def filter_recipes(self, action_recipes: List[ActionRecipe], behavioral_state: BehavioralState):
        return [self.filter_recipe(action_recipe, behavioral_state)
                for action_recipe in action_recipes]
