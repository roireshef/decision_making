from typing import Dict, List

import torch
from decision_making.src.rl_agent.agents.models.dual_inputs_actor_critic_model import DualInputsActorCriticModel
from decision_making.src.rl_agent.agents.models.utils import EnvUtils
from decision_making.src.rl_agent.global_types import STATE, ACTION_MASK
from ray.rllib.utils.typing import TensorType


class AttentionActorCriticModel(DualInputsActorCriticModel):
    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType) -> \
            (TensorType, List[TensorType]):
        """
        Forward an input state through the network/s
        :param input_dict: a dictionary containing a key named "obs" whose value is the environment state
        :param state: the hidden state (if RNN is used)
        :param seq_lens:
        :return:
        """
        ego_state, actors_state, actors_mask_shape = input_dict["obs"][STATE]
        action_mask = input_dict["obs"][ACTION_MASK]

        ego_features = self.ego_features_model(ego_state)

        # attention is using staticly-shaped tensor with mask (here the mask is created as a Bool tensor)
        # where True is value that should be used, False is masked out
        num_vehicles, max_num_vehicles = actors_mask_shape[:, 1], actors_state.shape[-1]
        actors_mask = self._get_mask(num_vehicles, max_num_vehicles)

        # key_padding_mask takes the inverse of the mask since attention is using True for masking out.
        actors_features = self.actors_features_model(actors_state, key_padding_mask=~actors_mask)

        shared_embeddings = self.shared_module(torch.cat((ego_features, actors_features), dim=1))

        # Logits of illegal (masked) actions is set to -infinity
        action_mask_logits = EnvUtils.mask_logits(action_mask)

        self.last_value = self.critic_net(shared_embeddings).squeeze(-1)

        return self.actor_net(shared_embeddings) + action_mask_logits, state

    def value_function(self) -> TensorType:
        return self.last_value

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
