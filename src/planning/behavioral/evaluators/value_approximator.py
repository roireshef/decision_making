from abc import ABCMeta
from logging import Logger

import six

from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState


@six.add_metaclass(ABCMeta)
class ValueApproximator:
    def __init__(self, logger: Logger):
        self.logger = logger

    def approximate(self, behavioral_state: BehavioralGridState) -> float:
        pass
