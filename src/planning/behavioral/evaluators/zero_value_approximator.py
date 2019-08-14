from logging import Logger

from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.evaluators.value_approximator import ValueApproximator
from typing import Any


class ZeroValueApproximator(ValueApproximator):
    def __init__(self, logger: Logger):
        super().__init__(logger)

    def approximate(self, behavioral_state: BehavioralGridState, goal: Any) -> float:
        return 0
