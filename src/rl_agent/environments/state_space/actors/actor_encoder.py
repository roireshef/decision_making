from abc import ABCMeta
from abc import abstractmethod
from typing import Tuple

import numpy as np
from gym.spaces import Box
from gym.spaces import Tuple as GymTuple
from decision_making.src.rl_agent.environments.state_space.common.data_objects import EgoCentricState, \
    EgoCentricActorState
from decision_making.src.rl_agent.environments.state_space.common.state_encoder import StateEncoder


class ActorEncoder(StateEncoder, metaclass=ABCMeta):
    """ Abstract class for encoding actors information.
        NOTE: Do not change public methods, implement only protected/private methodes """

    def encode(self, state: EgoCentricState, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        The external API for calling the encoder to encode the actors information from an EgoCentricState
        :param state: the input to the encoding process
        :param normalize: a flag determining either to normalize the state output or not. default: True.
        :return: a tuple of (normalized) state numpy array and mask numpy array
        """
        encoded, mask = self._encode(state)
        return self._normalize(encoded) if normalize else encoded, mask

    @property
    def observation_space(self) -> GymTuple:
        """ Actors state, Actors Mask shape """
        return GymTuple((
            self._actors_state_space,
            Box(low=0, high=np.iinfo(np.int32).max, shape=(len(self._actors_state_space.shape),), dtype=np.int)
        ))

    @property
    @abstractmethod
    def _actors_state_space(self) -> Box:
        """ Returns the shape and size(s) of the actors state (without the mask) """
        pass

    @property
    @abstractmethod
    def _num_actor_channels(self) -> int:
        """ The length of the 1D array per-actors channels (of self.actor_channels and self.empty_channels_template) """
        pass

    @abstractmethod
    def _encode(self, state: EgoCentricState) -> Tuple[np.ndarray, np.ndarray]:
        """
        The state encoding logic that transforms an EgoCentricState to encoded state array and mask array
        :param state: state to encode
        :return: encoded state array (numpy) and its mask shape (size of valid entries for all dimensions)
        """
        pass

    @abstractmethod
    def _normalize(self, encoded_state: np.ndarray) -> np.ndarray:
        """ The normalization logic that takes an encoded state and returns a normalized encoded state """
        pass

    @abstractmethod
    def _empty_channels_template(self, state: EgoCentricState) -> np.ndarray:
        """
        Return a 1D numpy array that represents channels for a single "empty" cell (with no actor)
        :param state: the full state used to extract any other relevant information for filling empty values
        :return: 1D numpy array
        """
        pass

    @abstractmethod
    def _actor_channels(self, actor_state: EgoCentricActorState, state: EgoCentricState) -> np.ndarray:
        """
        Implement a method that takes an actor's state and returns a 1D array of its channel-values
        :param actor_state: actor to encode
        :param state: the full state used to extract any other relevant information (baseline reductions, etc.)
        :return: 1D numpy array of channel values for the actor
        """
        pass


