import traceback
from abc import ABCMeta, abstractmethod
from logging import Logger
from typing import List, Optional

import six

import rte.python.profiler as prof
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState
from decision_making.src.planning.behavioral.data_objects import ActionSpec


@six.add_metaclass(ABCMeta)
class ActionSpecFilter:
    """
    Base class for filter implementations that act on ActionSpec and returns a boolean value that corresponds to
    whether the ActionSpec satisfies the constraint in the filter. All filters have to get as input ActionSpec
    (or one of its children) and  BehavioralState (or one of its children) even if they don't actually use them.
    """
    @abstractmethod
    def filter(self, action_specs: List[ActionSpec], behavioral_state: BehavioralState) -> List[ActionSpec]:
        pass

    def __str__(self):
        return self.__class__.__name__


class ActionSpecFiltering:
    """
    The gateway to execute filtering on one (or more) ActionSpec(s). From efficiency point of view, the filters
    should be sorted from the strongest (the one filtering the largest number of recipes) to the weakest.
    """
    def __init__(self, filters: Optional[List[ActionSpecFilter]], logger: Logger):
        self._filters: List[ActionSpecFilter] = filters or []
        self.logger = logger

    @prof.ProfileFunction()
    def filter_action_specs(self, action_specs: List[ActionSpec], behavioral_state: BehavioralState) -> List[bool]:
        for action_spec_filter in self._filters:
            try:
                action_specs = action_spec_filter.filter(action_specs, behavioral_state)
            except Exception:
                self.logger.warning('Exception during filtering at %s: %s', self.__class__.__name__, traceback.format_exc())
                return [spec is not None for spec in action_specs]
        return [spec is not None for spec in action_specs]
