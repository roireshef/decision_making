from decision_making.src.planning.behavioral.architecture.data_objects import ActionRecipe
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState


class RecipeFilter(object):
    def __init__(self, name, filtering_method):
        self.name = name
        self.filtering_method = filtering_method

    def __str__(self):
        return self.name


class RecipeFiltering:
    def __init__(self):
        self.filters = {}

    def add_filter(self, recipe_filter: RecipeFilter, is_active: bool) -> None:
        if recipe_filter not in self.filters:
            self.filters[recipe_filter] = is_active

    def activate_filter(self, recipe_filter: RecipeFilter, is_active: bool) -> None:
        if recipe_filter in self.filters:
            self.filters[recipe_filter] = is_active

    def filter_recipe(self, recipe: ActionRecipe, behavioral_state: BehavioralState) -> bool:
        for recipe_filter in self.filters.keys():
            if self.filters[recipe_filter]:
                result = recipe_filter.filtering_method(recipe, behavioral_state)
                if not result:
                    return False
        return True
