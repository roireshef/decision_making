from typing import List
from decision_making.src.planning.behavioral.architecture.components.filtering.action_spec_filtering import ActionSpecFiltering
from decision_making.src.planning.behavioral.architecture.data_objects import ActionSpec
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState


class ActionValidator:
    def __init__(self, action_spec_filtering: ActionSpecFiltering=None):
        self.action_spec_filtering = action_spec_filtering or ActionSpecFiltering()

    def validate_action(self, action_spec: ActionSpec, behavioral_state: BehavioralState) -> bool:
        return self.action_spec_filtering.filter_action_spec(action_spec, behavioral_state)

    def validate_actions(self, action_specs: List[ActionSpec], behavioral_state: BehavioralState):
        return [self.validate_action(action_spec, behavioral_state) for action_spec in
                action_specs]
