from logging import Logger

from decision_making.src.planning.behavioral.data_objects import NavigationGoal
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState


class ValueApproximator:
    def __init__(self, logger: Logger):
        self.logger = logger

    def approximate(self, behavioral_state: BehavioralGridState, goal: NavigationGoal) -> float:
        pass
