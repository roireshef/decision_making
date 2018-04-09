from decision_making.src.planning.behavioral.architecture.data_objects import ActionSpec
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState

# NOTE: All methods have to get as input ActionSpec (or one of its children) and  BehavioralState (or one of its
#       children) even if they don't actually use them.

# These methods are used as filters and are used to initialize ActionSpecFilter objects.


def filter_if_none(action_spec: ActionSpec, behavioral_state: BehavioralState) -> bool:
    if action_spec and behavioral_state:
        return True
    else:
        return False


def always_false(action_spec: ActionSpec, behavioral_state: BehavioralState) -> bool:
    if action_spec and behavioral_state:
        return False
    else:
        return True


