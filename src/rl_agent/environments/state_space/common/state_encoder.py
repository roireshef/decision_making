from abc import abstractmethod
from typing import Any

from gym.spaces import Space
from decision_making.src.rl_agent.environments.state_space.common.data_objects import EgoCentricState


class StateEncoder:
    @abstractmethod
    def encode(self, state: EgoCentricState, normalized: bool = True) -> Any:
        """
        Encode a raw EgoCentricState into a more structured and usable form.
        :param state: state to encode
        :param normalized: should encoding also normalize the output array
        :return: any type of encoded state
        """
        pass

    @property
    @abstractmethod
    def observation_space(self) -> Space:
        """ Returns the observation space for the environment that is using this state encoder. The observation-space
        is a Gym convention that represents the expected shape and structure of an observation """
        pass

    def __str__(self):
        return str("%s(%s)" % (self.__class__.__name__, self.__dict__))
