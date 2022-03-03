from typing import Dict
from typing import List

import gym
import numpy as np
import torch
from decision_making.src.rl_agent.agents.models.utils import ModelInits, ModelBuilder, EnvUtils
from decision_making.src.rl_agent.global_types import LAST_ACTION_IDX
from decision_making.src.rl_agent.global_types import STATE, ACTION_MASK
from decision_making.src.rl_agent.network_architectures import DUAL_IN_A3C_MODEL_CFG
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.tune.logger import logger
from torch import nn


# TODO: generalize to arbitrary number of input-blocks
class AttentionActorCriticModelV2(TorchModelV2, nn.Module):
    """ Adds last action embeddings """
    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str, architecture: str):

        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        architecture_structure = DUAL_IN_A3C_MODEL_CFG[architecture]

        ego_layers_structure = architecture_structure["ego_layers_structure"]
        actors_layers_structure = architecture_structure["actors_layers_structure"]
        last_action_layers_structure = list(map(lambda x: x.replace("{out}", str(num_outputs)),
                                                architecture_structure["last_action_layers_structure"]))
        shared_layers_structure = architecture_structure["shared_layers_structure"]
        actor_head_structure = list(map(lambda x: x.replace("{out}", str(num_outputs)),
                                        architecture_structure["actor_head_structure"]))
        critic_head_structure = architecture_structure["critic_head_structure"]

        ego_state_space, actors_state_space, _ = obs_space.original_space.spaces[STATE] \
            if hasattr(obs_space, 'original_space') else obs_space.spaces[STATE]

        # Define the the ego-features module:
        self.ego_features_model, ego_channels, ego_features = ModelBuilder.build(
            layer_config=ego_layers_structure,
            in_features=np.array(ego_state_space.shape)[1:],
            in_channels=ego_state_space.shape[0])

        # Define the the actors-features module:
        self.actors_features_model, actors_channels, actors_features = ModelBuilder.build(
            layer_config=actors_layers_structure,
            in_features=np.array(actors_state_space.shape)[1:],
            in_channels=actors_state_space.shape[0])

        # Define the the last-action applied module:
        self.last_action_model, last_action_channels, last_action_features = ModelBuilder.build(
            layer_config=last_action_layers_structure, in_features=1, in_channels=1)

        assert actors_channels == ego_channels, \
            "actors_features_model channels number (%s) differs from ego_features_model channels number (%s) " % \
            (actors_channels, ego_channels)

        assert np.int_(ego_features).shape == (), "only 1d channels are supported as ego_features"
        assert np.int_(actors_features).shape == (), "only 1d channels are supported as actors_features"
        assert np.int_(last_action_features).shape == (), "only 1d channels are supported as last_action_features"

        # Define the the shared layers module:
        self.shared_module, shared_channels, shared_features = ModelBuilder.build(
            shared_layers_structure, ego_features + actors_features + last_action_features, 1)

        # Define the the critic-head module:
        self.critic_net, critic_channels, critic_features = ModelBuilder.build(
            critic_head_structure, shared_features, shared_channels)

        # Define the the actor-head module:
        self.actor_net, actor_channels, actor_features = ModelBuilder.build(
            actor_head_structure, shared_features, shared_channels)

        logger.info(f"Initialized model {self.__class__.__name__} "
                    f"with total {self.num_trainable_params:,} trainable parameters")

        # initializations
        self.ego_features_model.apply(ModelInits.build_normal())
        self.actors_features_model.apply(ModelInits.build_xavier_relu())
        self.shared_module.apply(ModelInits.build_normal())
        self.critic_net.apply(ModelInits.build_normal())
        self.actor_net.apply(ModelInits.build_normal())

        self.last_value = None

    @property
    def num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType) -> \
            (TensorType, List[TensorType]):
        ego_state, actors_state, actors_mask_shape = input_dict["obs"][STATE]
        action_mask = input_dict["obs"][ACTION_MASK].to(ego_state.device)
        last_action_idx = input_dict["obs"][LAST_ACTION_IDX].to(ego_state.device)

        ego_features = self.ego_features_model(ego_state)

        # attention is using staticly-shaped tensor with mask (here the mask is created as a Bool tensor)
        # where True is value that should be used, False is masked out
        num_vehicles, max_num_vehicles = actors_mask_shape[:, 1], actors_state.shape[-1]
        actors_mask = self._get_mask(num_vehicles, max_num_vehicles)

        # key_padding_mask takes the inverse of the mask since attention is using True for masking out.
        actors_features = self.actors_features_model(actors_state, key_padding_mask=~actors_mask)

        # last action applied embeddings
        last_action_features = self.last_action_model(last_action_idx.long())

        shared_embeddings = self.shared_module(torch.cat((ego_features, actors_features, last_action_features), dim=1))

        # Logits of illegal (masked) actions is set to -infinity
        action_mask_logits = EnvUtils.mask_logits(action_mask)

        last_value = self.critic_net(shared_embeddings).squeeze(-1)
        self.last_value = last_value

        return self.actor_net(shared_embeddings) + action_mask_logits, state

    def value_function(self) -> TensorType:
        return self.last_value

    def import_from_h5(self, h5_file: str) -> None:
        pass

    @staticmethod
    def _get_mask(seq_lens: torch.LongTensor, max_seq_len: int):
        """
        Takes maximum sequence length (M; static size of a mask tensor) and a batch of (N) sequence lengths,
        and materializes a (NxM sized) mask tensor where on each row the first x values are True (valid) and the rest
        are False (masked out)
        :param seq_lens: tensor of indices that represents how many of the first elements should be True in the mask
        :param max_seq_len: maximal sequence length (number of columns in the mask)
        :return: NxM sized Bool tensor
        """
        return torch.arange(max_seq_len, device=seq_lens.device)[None, :] < seq_lens[:, None]
