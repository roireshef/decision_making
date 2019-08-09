from abc import ABCMeta, abstractmethod
from logging import Logger

import six

import rte.python.profiler as prof
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.messages.route_plan_message import RoutePlan

from typing import Any


@six.add_metaclass(ABCMeta)
class ValueApproximator:
    def __init__(self, logger: Logger):
        self.logger = logger

    @abstractmethod
    @prof.ProfileFunction()
    def approximate(self, behavioral_state: BehavioralGridState, route_plan: RoutePlan, goal: Any) -> float:
        pass
