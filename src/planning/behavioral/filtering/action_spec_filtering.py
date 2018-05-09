from typing import List, Callable

from decision_making.src.planning.behavioral.behavioral_state import BehavioralState
from decision_making.src.planning.behavioral.data_objects import ActionSpec


class ActionSpecFilter(object):
    def __init__(self, name, filtering_method: Callable[[ActionSpec, BehavioralState], bool]):
        self.name = name
        self.filtering_method = filtering_method

    def __str__(self):
        return self.name


class ActionSpecFiltering:
    def __init__(self, filters: List[ActionSpecFilter]=None):
        self._filters: List[ActionSpecFilter] = filters or []

    def filter_action_spec(self, action_spec: ActionSpec, behavioral_state: BehavioralState) -> bool:
        for action_spec_filter in self._filters:
            if not action_spec_filter.filtering_method(action_spec, behavioral_state):
                return False
        return True

    def filter_action_specs(self, action_specs: List[ActionSpec], behavioral_state: BehavioralState) -> List[bool]:
        return [self.filter_action_spec(action_spec, behavioral_state) for action_spec in action_specs]
