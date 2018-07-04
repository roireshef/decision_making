from abc import ABCMeta, abstractmethod
from logging import Logger

import six

import rte.python.profiler as prof
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.navigation.navigation_goal import NavigationGoal


@six.add_metaclass(ABCMeta)
class ValueApproximator:
    def __init__(self, logger: Logger):
        self.logger = logger

    @abstractmethod
    @prof.ProfileFunction()
    def approximate(self, behavioral_state: BehavioralGridState, goal: NavigationGoal) -> float:
        pass
