from decision_making.src.planning.behavioral.architecture.data_objects import ActionSpec
from decision_making.src.state.state import State


class ActionEvaluator:
    def evaluate(self, state: State, action: ActionSpec) -> float:
        pass
