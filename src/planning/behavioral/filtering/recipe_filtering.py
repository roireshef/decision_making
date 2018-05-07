import six
from abc import ABCMeta, abstractmethod
from typing import List

from decision_making.paths import Paths
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState
from decision_making.src.planning.behavioral.data_objects import ActionRecipe
from decision_making.src.planning.behavioral.filtering.recipe_filter_bank import FilterIfNone, \
    FilterActionsTowardsNonOccupiedCells, FilterActionsTowardBackAndParallelCells, FilterNonCalmActions, \
    FilterOvertakeActions, FilterBadExpectedTrajectory, FilterIfNoLane


@six.add_metaclass(ABCMeta)
class RecipeFilter:
    """
    Base class for filter implementations that act on ActionRecipe and returns a boolean value that corresponds to
    whether the ActionRecipe satisfies the constraint in the filter. All filters have to get as input ActionRecipe
    (or one of its children) and  BehavioralState (or one of its children) even if they don't actually use them.
    """
    @abstractmethod
    def filter(self, recipe: ActionRecipe, behavioral_state: BehavioralState) -> bool:
        pass

    def __str__(self):
        return self.__class__.__name__


class RecipeFiltering:
    """
    The gateway to execute filtering on one (or more) ActionRecipe(s). From efficiency point of view, the filters
    should be sorted from the strongest (the one filtering the largest number of recipes) to the weakest.
    """
    def __init__(self, filters: List[RecipeFilter] = None):
        self._filters: List[RecipeFilter] = filters or []

    def filter_recipe(self, recipe: ActionRecipe, behavioral_state: BehavioralState) -> bool:
        for recipe_filter in self._filters:
            if not recipe_filter.filter(recipe, behavioral_state):
                return False
        return True

    def filter_recipes(self, recipes: List[ActionRecipe], behavioral_state: BehavioralState) -> List[bool]:
        return [self.filter_recipe(recipe, behavioral_state) for recipe in recipes]


class DefaultRecipeFilters:
    dynamic_filters = [FilterIfNone(), FilterActionsTowardsNonOccupiedCells(),
                       FilterActionsTowardBackAndParallelCells(),
                       FilterOvertakeActions(), FilterNonCalmActions(),
                       FilterBadExpectedTrajectory(Paths.get_resource_path() + '/filters/bad_trajectory/predicates')]

    static_filters = [FilterIfNone(), FilterIfNoLane(), FilterNonCalmActions()]




