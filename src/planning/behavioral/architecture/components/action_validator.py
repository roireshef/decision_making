from typing import List
from decision_making.src.planning.behavioral.architecture.components.filtering import action_spec_filter_methods
from decision_making.src.planning.behavioral.architecture.components.filtering.action_spec_filtering import ActionSpecFiltering, \
    ActionSpecFilter
from decision_making.src.planning.behavioral.architecture.data_objects import ActionSpec
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState


class ActionValidator:
    def __init__(self, action_spec_filtering: ActionSpecFiltering):
        self.action_spec_filtering = action_spec_filtering
        self._init_filters()

    def _init_filters(self):
        self.action_spec_filtering.add_filter(
            ActionSpecFilter(name='FilterIfNone', filtering_method=action_spec_filter_methods.filter_if_none),
            is_active=True)
        self.action_spec_filtering.add_filter(
            ActionSpecFilter(name='AlwaysFalse', filtering_method=action_spec_filter_methods.always_false),
            is_active=False)

    def validate_action(self, action_spec: ActionSpec, behavioral_state: BehavioralState) -> bool:
        return self.action_spec_filtering.filter_action_spec(action_spec, behavioral_state)

    def validate_actions(self, action_specs: List[ActionSpec], behavioral_state: BehavioralState):
        return [self.validate_action(action_spec, behavioral_state) for action_spec in
                action_specs]
