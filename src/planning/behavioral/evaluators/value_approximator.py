from abc import ABCMeta, abstractmethod
from logging import Logger

import six

import rte.python.profiler as prof
from decision_making.src.planning.behavioral.state.behavioral_grid_state import BehavioralGridState
from typing import Any


@six.add_metaclass(ABCMeta)
class ValueApproximator:
    def __init__(self, logger: Logger):
        self.logger = logger

    @abstractmethod
    @prof.ProfileFunction()
    def approximate(self, behavioral_state: BehavioralGridState, goal: Any) -> float:
        pass
