from typing import List, Callable

from decision_making.src.planning.behavioral.behavioral_state import BehavioralState
from decision_making.src.planning.behavioral.data_objects import ActionRecipe


class RecipeFilter(object):
    def __init__(self, name: str, filtering_method: Callable[[ActionRecipe, BehavioralState], bool]):
        self.name = name
        self.filtering_method = filtering_method

    def __str__(self):
        return self.name


class RecipeFiltering:
    def __init__(self, filters: List[RecipeFilter]=None):
        self._filters: List[RecipeFilter] = filters or []

    def filter_recipe(self, recipe: ActionRecipe, behavioral_state: BehavioralState) -> bool:
        for recipe_filter in self._filters:
            if not recipe_filter.filtering_method(recipe, behavioral_state):
                return False
        return True

    def filter_recipes(self, recipes: List[ActionRecipe], behavioral_state: BehavioralState) -> List[bool]:
        return [self.filter_recipe(recipe, behavioral_state) for recipe in recipes]
