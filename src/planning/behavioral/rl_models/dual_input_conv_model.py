import numpy as np
import torch
from gym.spaces.tuple_space import Tuple as GymTuple
from ray.rllib.models.pytorch.model import TorchModel
from torch import nn
from typing import Dict, List


class DualInputConvModel(TorchModel):
    def __init__(self, obs_space, num_outputs, options):
        super().__init__(obs_space, num_outputs, options)

        custom_options = options["custom_options"]
        hidden_layer_size = custom_options["hidden_size"]

        ego_state_space, actors_state_space = obs_space.original_space.spaces \
            if hasattr(obs_space, 'original_space') else obs_space.spaces

        in_ego_features = np.prod(ego_state_space.shape)
        in_actors_features = np.prod(actors_state_space.shape)

        ego_embedding_size = hidden_layer_size

        # Define the model for the ego features:
        self.ego_features_model = nn.Sequential(
            nn.Linear(in_ego_features, ego_embedding_size),
            nn.ReLU())

        # Define the model for the actors features:
        self.actors_features_model = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=8, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU())

        in_features_size = 288  # ego_embedding_size + actors_embedding_size

        self.critic_net = nn.Sequential(
            nn.Linear(in_features_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, 1)
        )

        self.actor_net = nn.Sequential(
            nn.Linear(in_features_size, hidden_layer_size*2),
            nn.ReLU(),
            nn.Linear(hidden_layer_size*2, num_outputs)
        )

        self.apply(DualInputConvModel._init_weights)

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

        # Forward pass through the ego model:
        ego_features = self.ego_features_model(host_flat)
        # Forward pass through the actors model:
        actors_features = self.actors_features_model(env_state[1])
        all_features = torch.cat((ego_features, torch.reshape(actors_features, shape=[actors_features.shape[0], -1])), dim=1)

        return self.actor_net(all_features), None, self.critic_net(all_features).squeeze(-1), hidden_state
