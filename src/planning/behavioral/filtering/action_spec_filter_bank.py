from decision_making.src.planning.behavioral.behavioral_state import BehavioralState
from decision_making.src.planning.behavioral.data_objects import ActionSpec
from decision_making.src.planning.behavioral.filtering.action_spec_filtering import \
    ActionSpecFilter
from typing import List


class FilterIfNone(ActionSpecFilter):
    def filter(self, action_specs: List[ActionSpec], behavioral_state: BehavioralState) -> List[bool]:
        return [(action_spec and behavioral_state) is not None for action_spec in action_specs]
