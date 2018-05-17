from decision_making.src.planning.behavioral.behavioral_state import BehavioralState
from decision_making.src.planning.behavioral.data_objects import ActionSpec
from decision_making.src.planning.behavioral.filtering.action_spec_filtering import \
    ActionSpecFilter


class FilterIfNone(ActionSpecFilter):
    def filter(self, action_spec: ActionSpec, behavioral_state: BehavioralState) -> bool:
        return (action_spec and behavioral_state) is not None


class AlwaysFalse(ActionSpecFilter):
    def filter(self, action_spec: ActionSpec, behavioral_state: BehavioralState) -> bool:
        return False
