from decision_making.src.planning.behavioral.architecture.data_objects import ActionRecipe, ActionSpec
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState


class ActionSpecFilter(object):
    def __init__(self, name, filtering_method):
        self.name = name
        self.filtering_method = filtering_method

    def __str__(self):
        return self.name


class ActionSpecFiltering:
    def __init__(self):
        self.filters = {}

    def add_filter(self, action_spec_filter: ActionSpecFilter, is_active: bool) -> None:
        if action_spec_filter not in self.filters:
            self.filters[action_spec_filter] = is_active

    def activate_filter(self, action_spec_filter: ActionSpecFilter, is_active: bool) -> None:
        if action_spec_filter in self.filters:
            self.filters[action_spec_filter] = is_active

    def filter_action_spec(self, action_spec: ActionSpec, behavioral_state: BehavioralState) -> bool:
        for action_spec_filter in self.filters.keys():
            if self.filters[action_spec_filter]:
                result = action_spec_filter.filtering_method(action_spec, behavioral_state)
                if not result:
                    return False
        return True
