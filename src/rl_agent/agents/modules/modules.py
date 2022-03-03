import copy
from abc import abstractmethod
from typing import List

import torch
from decision_making.src.rl_agent.agents.modules.multihead_attention import MultiheadAttention
from torch import nn, Tensor
from torch.nn import functional as F
import torch


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Transpose(nn.Module):
    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


class ResBlock(nn.Module):
    def __init__(self, pre_concat: nn.Sequential, post_concat: nn.Sequential):
        """
        Residual block
        :param pre_concat: nn.Sequential that is applied to inputs, and then concatenated to them
        :param post_concat: nn.Sequential that is applied to the concatenation of pre_concat and inputs
        """
        super().__init__()
        self.pre_residual = pre_concat
        self.post_residual = post_concat

    def forward(self, x):
        return self.post_residual(self.pre_residual(x) + x)


class MaxLastDim(nn.Module):
    """ Max-pools over the last dimension. If last dimension is the entity dimension, then we get entity-wise global
    feature vector """
    @staticmethod
    def forward(x):
        return F.max_pool1d(x, kernel_size=x.shape[-1])


class ConcatWithMaxedLastDim(nn.Module):
    """ Max-pools over the last dimension and concatenates resulting vector with the original tensor """
    @staticmethod
    def forward(x):
        global_vector = MaxLastDim.forward(x)
        return torch.cat((x, global_vector.repeat(1, 1, x.shape[-1])), 1)


class MaskableForwardMixin:
    @abstractmethod
    def forward(self, src, src_mask=None, src_key_padding_mask=None) -> Tensor:
        """
        forward inference method signature that takes masks (suitable for attention blocks)
        :param src: input tensor
        :param src_mask: MultiheadAttention's src_mask
        :param src_key_padding_mask: MultiheadAttention's src_key_padding_mask
        :return: output tensor
        """
        pass


class MaskableSequential(MaskableForwardMixin, nn.Sequential):
    """ Container for sequence of nn.Modules (similar to nn.Sequential) that are implementing either .forward(x) or
    .forward(x, mask, key_padding_mask) where masks are propagated """
    def forward(self, x: Tensor, mask=None, key_padding_mask=None) -> Tensor:
        for module in self:
            if issubclass(type(module), MaskableForwardMixin):
                x = module(x, mask=mask, key_padding_mask=key_padding_mask)
            else:
                x = module(x)

        return x


class MultiHeadSelfAttention(MaskableForwardMixin, nn.Module):
    def __init__(self, d_model, nhead=4, dim_feedforward=256, activation=F.relu, mask_value: float = 0):
        """
        Self (q=k=v) Multihead Attention operator, followed by two FC layers and activation in between
        :param d_model: the size of the embedding dimension (nhead * n_k)
        :param nhead: number of heads to split d_model into groups of sub-embeddings
        :param dim_feedforward: the layer size that follows the attention operator
        :param activation: the activation used after the layer that follows the attention operator
        """
        super().__init__()

        self.self_attn = MultiheadAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = activation
        self.mask_value = mask_value

    def forward(self, src, mask=None, key_padding_mask=None) -> Tensor:
        # pytorch's MultiheadAttention requires sequence-major tensor [seq, batch, embeddings]
        seq_major = src.transpose(0, 1)
        att_out, att_weights = self.self_attn(seq_major, seq_major, seq_major, attn_mask=mask,
                                              key_padding_mask=key_padding_mask)

        # transpose back to [batch, seq, embeddings] + skip connection over attention
        att_out = att_out.transpose(0, 1) + src

        # two fully connected layers with activation + skip connection over fc
        fc_out = self.linear2(self.activation(self.linear1(att_out))) + att_out

        # re-mask self-attention queries of masked out keys (key_padding_mask is only applied to keys, not to queries)
        out_masked = self._mask_attention_results(fc_out, key_padding_mask)

        return out_masked

    def _mask_attention_results(self, tensor, mask):
        """
        Fills <self.mask_value> in cells of results that are True in mask
        :param tensor: tensor of shape [Batch, Seq, Emb]
        :param mask: tensor of shape [B, S]
        :return: tensor of shape [B, S, E] with values where mask==True replaced by <self.mask_value>
        """
        if mask is not None:
            tensor = tensor.masked_fill(mask.unsqueeze(-1), self.mask_value)

        return tensor


class SelfAttentionEncoder(MaskableForwardMixin, nn.Module):
    def __init__(self, layers: List[MultiHeadSelfAttention]):
        super().__init__()
        self.attn_layers = MaskableSequential(*layers)

    @classmethod
    def from_layer(cls, layer: MultiHeadSelfAttention, num_layers: int):
        return cls([copy.deepcopy(layer) for _ in range(num_layers)])

    def forward(self, src, mask=None, key_padding_mask=None) -> Tensor:
        """
        Takes src tensor of shape [batch, emb, seq] and applies attention over entities in the seq dimension, each
        layer is a MultiHeadSelfAttention padded seq positions are masked to zero
        :param src: tensor of shape [batch, emb, seq]
        :param mask: passed to MultiHeadSelfAttention (See description there)
        :param key_padding_mask: bool tensor of shape  [batch, seq] that represents which entities in seq will be taken
        into account in the attention operator. False values will be masked out of the results as well (will have zero
        vectors)
        :return:  [batch, emb, seq] tensor where emb dimension is now the output of attention operator between seq
        entities, across all the batch (attention is applied for each batch position independently)
        """

        transposed = src.transpose(-1, -2)  # [batch, emb, seq] -> [batch, seq, emb]

        attn_output = transposed
        for layer in self.attn_layers:
            attn_output = layer(attn_output, mask, key_padding_mask)

        transposed_back = attn_output.transpose(-1, -2)  # [batch, seq, emb] -> [batch, emb, seq]

        return transposed_back


class NullableEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim):
        """
        Embedding layer that takes -1 for null embedding (zeros, non-trainable)
        :param num_embeddings: number of embedding keys [0...(n-1)] (n is the null embedding)
        :param embedding_dim: dimensionality of embedding vectors
        """
        super().__init__(num_embeddings+1, embedding_dim, padding_idx=-1)

    def forward(self, input: Tensor) -> Tensor:
        return super().forward(torch.arange(self.num_embeddings, device=self.weight.device)[input])
