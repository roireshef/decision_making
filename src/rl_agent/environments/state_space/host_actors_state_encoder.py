from typing import NamedTuple

import numpy as np
from gym.spaces import Tuple as GymTuple
from decision_making.src.rl_agent.environments.state_space.actors.actor_encoder import ActorEncoder
from decision_making.src.rl_agent.environments.state_space.common.data_objects import EgoCentricState
from decision_making.src.rl_agent.environments.state_space.common.state_encoder import StateEncoder
from decision_making.src.rl_agent.environments.state_space.host.host_encoder import HostEncoder


class EncodedState(NamedTuple):
    host_state: np.ndarray
    actors_state: np.ndarray
    actors_mask: np.ndarray


class HostActorsStateEncoder(StateEncoder):
    """ Encodes an EgoCentricState (host and actors) into a gym.Tuple[gym.Box, gym.Box] """

    def __init__(self, host_encoder: HostEncoder, actors_encoder: ActorEncoder):
        self.host_encoder = host_encoder
        self.actors_encoder = actors_encoder

    def encode(self, state: EgoCentricState, normalize: bool = True) -> EncodedState:
        """
        Encode an EgoCentricState (host and actors) into a gym.Tuple[gym.Box, gym.Box]
        :param state: state to encode
        :param normalize: should encoding also normalize the output array
        :return: EncodedState of host tensor and actors tensor
        """
        actors_state, actors_mask = self.actors_encoder.encode(state, normalize)
        host_state = self.host_encoder.encode(state, normalize)
        return EncodedState(host_state, actors_state, actors_mask)

    @property
    def observation_space(self) -> GymTuple:
        return GymTuple((self.host_encoder.observation_space, *self.actors_encoder.observation_space))
