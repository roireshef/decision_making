from decision_making.src.planning.behavioral.architecture.data_objects import ActionRecipe
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState
from typing import List


class RecipeFilter(object):
    def __init__(self, name, filtering_method):
        self.name = name
        self.filtering_method = filtering_method

    def __str__(self):
        return self.name


class RecipeFiltering:
    def __init__(self, filters: List[RecipeFilter]=None):
        if filters is None:
            filters = []
        self._filters = filters

    def filter_recipe(self, recipe: ActionRecipe, behavioral_state: BehavioralState) -> bool:
        for recipe_filter in self._filters:
            result = recipe_filter.filtering_method(recipe, behavioral_state)
            if not result:
                return False
        return True

    def filter_recipes(self, recipes: List[ActionRecipe], behavioral_state: BehavioralState) -> List[bool]:
        return [self.filter_recipe(recipe, behavioral_state) for recipe in recipes]
