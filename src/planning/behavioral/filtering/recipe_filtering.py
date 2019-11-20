import six
from abc import ABCMeta, abstractmethod

from decision_making.src.planning.behavioral.state.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionRecipe
from logging import Logger
from typing import List, Optional


@six.add_metaclass(ABCMeta)
class RecipeFilter:
    """
    Base class for filter implementations that act on ActionRecipe and returns a boolean value that corresponds to
    whether the ActionRecipe satisfies the constraint in the filter. All filters have to get as input ActionRecipe
    (or one of its children) and  BehavioralGridState (or one of its children) even if they don't actually use them.
    """

    @abstractmethod
    def filter(self, recipes: List[ActionRecipe], behavioral_state: BehavioralGridState, logger: Optional[Logger] = None) -> List[bool]:
        """
        Filters an ActionRecipe based on the state of ego and nearby vehicles (BehavioralGridState).
        :param recipes: an object representing the semantic action to be considered
        :param behavioral_state: semantic behavioral state, containing the semantic grid
        :return: A boolean result, True if recipe is valid and false if filtered
        """
        pass

    def __str__(self):
        return self.__class__.__name__


class RecipeFiltering:
    """
    The gateway to execute filtering on one (or more) ActionRecipe(s). From efficiency point of view, the filters
    should be sorted from the strongest (the one filtering the largest number of recipes) to the weakest.
    """

    def __init__(self, filters: Optional[List[RecipeFilter]], logger: Logger):
        self._filters: List[RecipeFilter] = filters or []
        self.logger = logger

    def filter_recipes(self, recipes: List[ActionRecipe], behavioral_state: BehavioralGridState) -> List[bool]:
        """
        Filters a list of 'ActionRecipe's based on the state of ego and nearby vehicles (BehavioralGridState).
        :param recipes: A list of objects representing the semantic actions to be considered
        :param behavioral_state: semantic behavioral state, containing the semantic grid
        :return: A boolean List , True where the respective recipe is valid and false where it is filtered
        """
        mask = [True for i in range(len(recipes))]
        for recipe_filter in self._filters:
            mask = recipe_filter.filter(recipes, behavioral_state, self.logger)
            recipes = [recipes[i] if mask[i] else None for i in range(len(recipes))]
        return mask

    def filter_recipe(self, recipe: ActionRecipe, behavioral_state: BehavioralGridState) -> bool:
        """
        Filters an 'ActionRecipe's based on the state of ego and nearby vehicles (BehavioralGridState).
        :param recipe: An object representing the semantic actions to be considered
        :param behavioral_state: semantic behavioral state, containing the semantic grid
        :return: A boolean , True where the recipe is valid and false where it is filtered
        """
        return self.filter_recipes([recipe], behavioral_state)[0]
