import numpy as np
import rte.python.profiler as prof
import six
from abc import ABCMeta, abstractmethod

from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionSpec
from logging import Logger
from typing import List, Optional
from itertools import compress

@six.add_metaclass(ABCMeta)
class ActionSpecFilter:
    """
    Base class for filter implementations that act on ActionSpec and returns a boolean value that corresponds to
    whether the ActionSpec satisfies the constraint in the filter. All filters have to get as input ActionSpec
    (or one of its children) and  BehavioralGridState (or one of its children) even if they don't actually use them.
    """
    @abstractmethod
    def filter(self, action_specs: List[ActionSpec], behavioral_state: BehavioralGridState) -> List[bool]:
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

    def filter_action_specs(self, action_specs: List[ActionSpec], behavioral_state: BehavioralGridState) -> List[bool]:
        """
        Filters a list of 'ActionSpec's based on the state of ego and nearby vehicles (BehavioralGridState).
        :param action_specs: A list of objects representing the specified actions to be considered
        :param behavioral_state: semantic behavioral state, containing the semantic grid
        :return: A boolean List , True where the respective action_spec is valid and false where it is filtered
        """
        mask = np.full(shape=len(action_specs), fill_value=True, dtype=np.bool)
        for action_spec_filter in self._filters:
            if ~np.any(mask):
                break
            # list of only valid action specs
            valid_action_specs = list(compress(action_specs, mask))
            # a mask only on the valid action specs
            current_mask = action_spec_filter.filter(valid_action_specs, behavioral_state)
            # use the reduced mask to update the original mask (that contains all initial actions specs given)
            mask[mask] = current_mask
        return mask.tolist()

    @prof.ProfileFunction()
    def filter_action_spec(self, action_spec: ActionSpec, behavioral_state: BehavioralGridState) -> bool:
        """
        Filters an 'ActionSpec's based on the state of ego and nearby vehicles (BehavioralGridState).
        :param action_spec: An object representing the specified actions to be considered
        :param behavioral_state: semantic behavioral state, containing the semantic grid
        :return: A boolean , True where the action_spec is valid and false where it is filtered
        """
        return self.filter_action_specs([action_spec], behavioral_state)[0]

