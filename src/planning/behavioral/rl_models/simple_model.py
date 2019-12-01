import numpy as np
import torch
from gym.spaces.tuple_space import Tuple as GymTuple
from ray.rllib.models.pytorch.model import TorchModel
from torch import nn
from typing import Dict, List


class SimpleModel(TorchModel):
    def __init__(self, obs_space, num_outputs, options):
        super().__init__(obs_space, num_outputs, options)

        custom_options = options["custom_options"]
        hidden_layer_size = custom_options["hidden_size"]

        in_features = np.prod(obs_space.shape)

        self.critic_net = nn.Sequential(
            nn.Linear(in_features, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, 1)
        )

        self.actor_net = nn.Sequential(
            nn.Linear(in_features, hidden_layer_size*2),
            nn.ReLU(),
            nn.Linear(hidden_layer_size*2, num_outputs)
        )

        self.apply(SimpleModel._init_weights)

    @staticmethod
    def _init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, mean=0., std=0.1)
            nn.init.constant_(m.bias, 0.)

    @staticmethod
    def from_env_state(env_state: GymTuple) -> GymTuple:
        """
        Convert state from environment state to this model state
        :param env_state:
        :return:
        """
        host_state, actors_state = env_state
        host_flat = torch.flatten(host_state, start_dim=1)
        actors_flat = torch.flatten(actors_state, start_dim=1)
        return host_flat, actors_flat

    def _forward(self, input_dict: Dict, hidden_state: List = []) -> (torch.Tensor, None, torch.Tensor, List):
        """
        Forward an input state through the network/s
        :param input_dict: a dictionary containing a key named "obs" whose value is the environment state
        :param hidden_state: the hidden state (if RNN is used)
        :return:
        """
        env_state = input_dict["obs"]
        host_flat, actors_flat = self.from_env_state(env_state)
        state = torch.cat((host_flat, actors_flat), dim=1)
        return self.actor_net(state), None, self.critic_net(state).squeeze(-1), hidden_state
