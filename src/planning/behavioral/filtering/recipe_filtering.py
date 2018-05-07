from abc import ABCMeta, abstractmethod
from typing import List, Callable

from decision_making.src.global_constants import BP_JERK_S_JERK_D_TIME_WEIGHTS
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState
from decision_making.src.planning.behavioral.data_objects import ActionRecipe, DynamicActionRecipe, \
    RelativeLongitudinalPosition, ActionType, RelativeLane, AggressivenessLevel
from decision_making.src.planning.utils.math import Math
from decision_making.src.planning.utils.optimal_control.quintic_poly_formulas import v_0_grid, a_0_grid, s_T_grid, \
    v_T_grid
from mapping.src.service.map_service import MapService
import six


@six.add_metaclass(ABCMeta)
class RecipeFilter(object):
    @abstractmethod
    def filter(self, recipe: DynamicActionRecipe, behavioral_state: BehavioralGridState) -> bool:
        pass

    def __str__(self):
        return self.__class__.__name__


class RecipeValidator:
    def __init__(self, filters: List[RecipeFilter] = None):
        self._filters: List[RecipeFilter] = filters or []

    def filter_recipe(self, recipe: ActionRecipe, behavioral_state: BehavioralState) -> bool:
        for recipe_filter in self._filters:
            if not recipe_filter.filtering_method(recipe, behavioral_state):
                return False
        return True

    def filter_recipes(self, recipes: List[ActionRecipe], behavioral_state: BehavioralState) -> List[bool]:
        return [self.filter_recipe(recipe, behavioral_state) for recipe in recipes]


class DynamicRecipeFilterBank:
    @staticmethod
    def get_all() -> List[RecipeFilter]:
        return [RecipeFilter(name='filter_if_none', filtering_method=DynamicRecipeFilterBank.filter_if_none),
                RecipeFilter(name="filter_actions_towards_non_occupied_cells",
                             filtering_method=DynamicRecipeFilterBank.filter_actions_towards_non_occupied_cells),
                RecipeFilter(name="filter_actions_toward_back_and_parallel_cells",
                             filtering_method=DynamicRecipeFilterBank.filter_actions_toward_back_and_parallel_cells),
                RecipeFilter(name="filter_over_take_actions",
                             filtering_method=DynamicRecipeFilterBank.filter_over_take_actions),
                RecipeFilter(name='filter_non_calm_actions',
                             filtering_method=DynamicRecipeFilterBank.filter_non_calm_actions)]







