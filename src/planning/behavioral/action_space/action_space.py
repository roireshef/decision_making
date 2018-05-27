import itertools
from abc import abstractmethod
from collections import defaultdict
from logging import Logger
from typing import List, Optional, Type

import rte.python.profiler as prof
from decision_making.src.exceptions import raises
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState
from decision_making.src.planning.behavioral.data_objects import ActionRecipe
from decision_making.src.planning.behavioral.data_objects import ActionSpec
from decision_making.src.planning.behavioral.filtering.recipe_filtering import RecipeFiltering


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
        self._recipe_filtering = recipe_filtering or RecipeFiltering(None, logger)

    @property
    def action_space_size(self) -> int:
        return len(self._recipes)

    @property
    def recipes(self) -> List[ActionRecipe]:
        """returns all recipes generated by an ActionSpace implementation"""
        return self._recipes

    @property
    @abstractmethod
    def recipe_classes(self) -> List[Type]:
        """lists all recipe class types that an ActionSpace implementation can create"""
        pass

    @prof.ProfileFunction()
    def filter_recipe(self, action_recipe: ActionRecipe, behavioral_state: BehavioralState) -> bool:
        """
        For a given recipe (and a state), returns true if the recipe passes all filters in self._recipe_filtering
        :param action_recipe: recipe to validate
        :param behavioral_state: current state of the world
        :return: true if recipe passes all filters, false otherwise
        """
        return self._recipe_filtering.filter_recipe(action_recipe, behavioral_state)

    @prof.ProfileFunction()
    def filter_recipes(self, action_recipes: List[ActionRecipe], behavioral_state: BehavioralState) -> List[bool]:
        """
        For a given list of recipes (and a state) - for each recipe, returns true if the recipe passes all filters
        in self._recipe_filtering
        :param action_recipes: recipes to validate
        :param behavioral_state: current state of the world
        :return: list - true where a recipe passes all filters, false otherwise
        """
        return self._recipe_filtering.filter_recipes(action_recipes, behavioral_state)

    def specify_goal(self, action_recipe: ActionRecipe, behavioral_state: BehavioralGridState) -> Optional[ActionSpec]:
        return self.specify_goals([action_recipe], behavioral_state)[0]

    @abstractmethod
    def specify_goals(self, action_recipes: List[ActionRecipe], behavioral_state: BehavioralGridState) -> List[Optional[ActionSpec]]:
        """
        This method's purpose is to specify the enumerated actions (recipes) that the agent can take.
        Each semantic action (ActionRecipe) is translated into a terminal state specification (ActionSpec).
        :param action_recipes: an enumerated semantic action [ActionRecipe].
        :param behavioral_state: a Frenet state of ego at initial point
        :return: semantic action specification [ActionSpec] or [None] if recipe can't be specified.
        """
        pass


class ActionSpaceContainer(ActionSpace):
    def __init__(self, logger: Logger, action_spaces: List[ActionSpace]):
        """
        This class acts as a container for action_spaces and is responsible for filtering and specifying each recipe with
        the respective class (e.g. StaticActionSpace or DynamicActionSpace)
        :param logger: dedicated logger implementation
        :param action_spaces: a list of various implementations of the interface ActionSpace, containing recipes (List[ActionSpace])
        """
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

    @raises(NotImplemented)
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

    @raises(NotImplemented)
    def filter_recipe(self, action_recipe: ActionRecipe, behavioral_state: BehavioralState) -> bool:
        return self._recipe_handler[action_recipe.__class__].filter_recipe(action_recipe, behavioral_state)

    # TODO: figure out how to remove the for loop for better efficiency and stay consistent with ordering
    @raises(NotImplemented)
    def filter_recipes(self, action_recipes: List[ActionRecipe], behavioral_state: BehavioralState):
        return [self.filter_recipe(action_recipe, behavioral_state)
                for action_recipe in action_recipes]
