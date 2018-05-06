from decision_making.src.planning.behavioral.behavioral_state import BehavioralState
from decision_making.src.planning.behavioral.data_objects import ActionSpec
from decision_making.src.planning.behavioral.filtering import \
    ActionSpecFilter


# NOTE: All methods have to get as input ActionSpec (or one of its children) and  BehavioralState (or one of its
#       children) even if they don't actually use them.

# These methods are used as filters and are used to initialize ActionSpecFilter objects.


def filter_if_none(action_spec: ActionSpec, behavioral_state: BehavioralState) -> bool:
    return (action_spec and behavioral_state) is not None


def always_false(action_spec: ActionSpec, behavioral_state: BehavioralState) -> bool:
    return False


# Filter list definition
action_spec_filters = [ActionSpecFilter(name='filter_if_none', filtering_method=filter_if_none)]