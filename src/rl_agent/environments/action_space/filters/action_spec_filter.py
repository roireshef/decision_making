from abc import ABCMeta, abstractmethod
from typing import List

from decision_making.src.planning.types import BoolArray
from decision_making.src.rl_agent.environments.action_space.common.data_objects import RLActionSpec
from decision_making.src.rl_agent.environments.state_space.common.data_objects import EgoCentricState


class ActionSpecFilter(metaclass=ABCMeta):
    @abstractmethod
    def filter(self, action_specs: List[RLActionSpec], state: EgoCentricState) -> BoolArray:
        """
        The abstract method to be implemented by user for filtering actions given current state's information
        :param action_specs: the list of actions to filter
        :param state: the state of environment
        :return: 1d numpy bool array (same length as action_specs) - true if action is valid, false otherwise
        """
        pass

    def __str__(self):
        return "%s(%s)" % (self.__class__.__name__, self.__dict__)
