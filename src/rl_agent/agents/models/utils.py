from typing import List, Union, Dict, Callable

import numpy as np
import torch
from gym.spaces import Tuple as GymTuple
from math import floor
from decision_making.src.rl_agent.agents.modules.modules import MultiHeadSelfAttention, MaskableSequential, Transpose, \
    SelfAttentionEncoder, NullableEmbedding, Flatten, ResBlock, ConcatWithMaxedLastDim, \
    MaxLastDim
from ray.rllib.agents.trainer import logger
from torch import nn


class ModelBuilder:
    conv_param_keys = {
        "K": "kernel_size",
        "P": "padding",
        "D": "dilation",
        "S": "stride"
    }

    attention_param_keys = {
        "M": "d_model",
        "H": "nhead",
        "N": "dim_feedforward"
    }

    embedding_param_keys = {
        "E": "num_embeddings"
    }

    @staticmethod
    def conv_output_size(input_size: Union[int, np.ndarray], kernel_size: Union[int, np.ndarray],
                         stride: Union[int, np.ndarray], dilation: Union[int, np.ndarray],
                         padding: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
        """
        Utility function for computing output of convolutions. All inputs can be integers for the nn.Conv1d case, or 1d
        numpy arrays (2 values each) for nn.Conv2d
        """
        return np.floor(((input_size + (padding * 2) - (dilation * (kernel_size - 1)) - 1) / stride) + 1)

    @staticmethod
    def parse_params(params_string: str, params_shortcuts: Dict, val_func: Callable = lambda x: x) -> Dict:
        return {params_shortcuts[param[0]]: val_func(param[1:]) for param in params_string}

    @staticmethod
    def build(layer_config: List[str], in_features: Union[int, np.ndarray], in_channels: int = 1) \
            -> (MaskableSequential, int, Union[int, np.ndarray]):
        """
        This utility function creates an nn.Sequential with the configuration given and the sizes of feature dimension
        and channel dimension for its inputs. <layer_config> is a list of layer configs - each has the format:
        <layer_type>_<layer_parameters>, with:
        * Activations: layer_type = "A", layer_parameters = activation class name
        * Flatten layer: layer_type = "F"
        * FullyConnected (Linear): layer_type = "FC", layer_parameters = number of output features
        * Conv1d: layer_type = "C1", layer_parameters = number of out channels (optional parameters
          delimited by '|' with shortcuts from ModelUtils.layer_config_shortcuts)
        * Conv2d: layer_type = "C2", layer_parameters = number of out channels (optional parameters
          delimited by '|' with shortcuts from ModelUtils.layer_config_shortcuts)
        * MaxPool1d layer layer_type = "M1" (optional parameters delimited by '|' with shortcuts from
          ModelUtils.layer_config_shortcuts)
        * MaxPool2d layer layer_type = "M2" (optional parameters delimited by '|' with shortcuts from
          ModelUtils.layer_config_shortcuts)
        * BatchNorm1d layer layer layer_type = "BN1"

        Example:
            layer_config = ["C1_32|K5", "A_ReLU", "C1_64|K3", "A_ReLU", "F", "FC_64", "A_Tanh"]
            will create the following layers:
                * nn.Conv1D with 32 output channels and kernel of size 5
                * nn.ReLU
                * nn.Conv1D with 64 output channels and kernel of size 3
                * nn.ReLU
                * Flatten layer
                * nn.Linear with 64 output features
                * nn.Tanh

        :param layer_config: the configuration list as explained above
        :param in_features: the size of the features dimension (an array if 2D)
        :param in_channels: the size of the channels dimension
        :return: the MaskableSequential it builds, num of output channels, num of output features (an array if 2D)
        """
        last_channels = in_channels
        last_features = in_features.item() if isinstance(in_features, np.ndarray) and len(in_features) == 1 else in_features

        layers = []
        for i, layer in enumerate(layer_config):
            if layer.__contains__("_"):
                layer_type, layer_params = layer.split("_")
            else:
                layer_type = layer

            # Activation layer
            if layer_type == "A":
                layers.append(eval("nn.%s()" % layer_params))

            # Fully-connected layer
            elif layer_type == "FC":
                assert last_channels == 1, "layer %s is Linear but in channels (%s) > 1 in %s" % (
                    i, last_channels, layer_config)
                out_features = int(layer_params)
                layers.append(nn.Linear(last_features, out_features))
                last_features = out_features

            # Fully-connected layer with skip connection based on ResBlock
            elif layer_type == "SkipFC":
                assert last_channels == 1, "layer %s is Linear but in channels (%s) > 1 in %s" % (
                    i, last_channels, layer_config)
                pre_residual_model, _, _ = ModelBuilder.build(
                    ["FC_%d" % last_features, "A_ReLU", "FC_%d" % last_features], last_features, 1)
                post_residual_model, _, _ = ModelBuilder.build(["A_ReLU"], last_features, 1)
                layers.append(ResBlock(pre_residual_model, post_residual_model))

            # Fully-connected layer with skip connection based on ResBlock
            elif layer_type == "SingleSkipFC":
                assert last_channels == 1, "layer %s is Linear but in channels (%s) > 1 in %s" % (
                    i, last_channels, layer_config)
                pre_residual_model, _, _ = ModelBuilder.build(
                    ["A_ReLU", "FC_%d" % last_features], last_features, 1)
                post_residual_model, _, _ = ModelBuilder.build(["A_ReLU"], last_features, 1)
                layers.append(ResBlock(pre_residual_model, post_residual_model))

            # Nullable Embedding layer (for null value use value -1, otherwise use 0..n)
            elif layer_type == "EMB":
                out_features_str, *other_params = layer_params.split('|')
                out_features = int(out_features_str)
                param_dict = ModelBuilder.parse_params(other_params, ModelBuilder.embedding_param_keys, int)
                layers.append(NullableEmbedding(embedding_dim=out_features, **param_dict))
                last_features = out_features
                last_channels = 1

            # Conv1d layer
            elif layer_type == "C1":
                out_channels_str, *other_params = layer_params.split('|')
                out_channels = int(out_channels_str)
                param_dict = ModelBuilder.parse_params(other_params, ModelBuilder.conv_param_keys, int)
                new_layer = nn.Conv1d(in_channels=last_channels, out_channels=out_channels, **param_dict)
                layers.append(new_layer)
                last_channels = out_channels
                last_features = ModelBuilder.conv_output_size(last_features, new_layer.kernel_size[0],
                                                              new_layer.stride[0], new_layer.dilation[0],
                                                              new_layer.padding[0])

            # Conv2d layer
            elif layer_type == "C2":
                out_channels_str, *other_params = layer_params.split('|')
                out_channels = int(out_channels_str)
                param_dict = ModelBuilder.parse_params(other_params, ModelBuilder.conv_param_keys, eval)
                new_layer = nn.Conv2d(in_channels=last_channels, out_channels=out_channels, **param_dict)
                layers.append(new_layer)
                last_channels = out_channels
                last_features = ModelBuilder.conv_output_size(last_features, np.array(new_layer.kernel_size),
                                                              np.array(new_layer.stride),
                                                              np.array(new_layer.dilation),
                                                              np.array(new_layer.padding))

            # ResBlock (1d) layer without BatchNorm1d
            elif layer_type == "RB1":
                out_channels_str, *_ = layer_params.split('|')
                pre_residual_model, pre_residual_channels, pre_residual_features = ModelBuilder.build(
                    ["C1_" + layer_params, "A_ReLU", "C1_" + layer_params], last_features, last_channels)
                post_residual_model, post_residual_channels, post_residual_features = ModelBuilder.build(
                    ["A_ReLU"], pre_residual_features, pre_residual_channels)
                layers.append(ResBlock(pre_residual_model, post_residual_model))
                last_channels = post_residual_channels
                last_features = post_residual_features

            # ResBlock (2d) layer without BatchNorm1d
            elif layer_type == "RB2":
                out_channels_str, *_ = layer_params.split('|')
                pre_residual_model, pre_residual_channels, pre_residual_features = ModelBuilder.build(
                    ["C2_" + layer_params, "A_ReLU", "C2_" + layer_params], last_features, last_channels)
                post_residual_model, post_residual_channels, post_residual_features = ModelBuilder.build(
                    ["A_ReLU"], pre_residual_features, pre_residual_channels)
                layers.append(ResBlock(pre_residual_model, post_residual_model))
                last_channels = post_residual_channels
                last_features = post_residual_features

            # ResBlock (1d) layer with BatchNorm1d after each convolution
            elif layer_type == "RB1BN":
                out_channels_str, *_ = layer_params.split('|')
                pre_residual_model, pre_residual_channels, pre_residual_features = ModelBuilder.build(
                    ["C1_" + layer_params, "BN1", "A_ReLU", "C1_" + layer_params, "BN1"], last_features, last_channels)
                post_residual_model, post_residual_channels, post_residual_features = ModelBuilder.build(
                    ["A_ReLU"], pre_residual_features, pre_residual_channels)
                layers.append(ResBlock(pre_residual_model, post_residual_model))
                last_channels = post_residual_channels
                last_features = post_residual_features

            # Self Attention (1d) layer
            elif layer_type == "SA":
                params = layer_params.split('|')
                param_dict = ModelBuilder.parse_params(params, ModelBuilder.attention_param_keys, int)
                assert last_channels == param_dict['d_model'], \
                    "layer %d is Self Attention with d_model (%d) != in channels (%d) " % (i, param_dict['d_model'], last_channels)
                layers.append(MultiHeadSelfAttention(**param_dict))

            # Self Attention (1d) Encoder (sequence of self attention layer with trasnposing before and after)
            elif layer_type == "SAENC":
                num_layers, *params = layer_params.split('|')
                param_dict = ModelBuilder.parse_params(params, ModelBuilder.attention_param_keys, int)
                assert last_channels == param_dict['d_model'], \
                    "layer %d is SelfAttentionEncoder with d_model (%d) != in channels (%d) " % (i, param_dict['d_model'], last_channels)
                layer, *_ = ModelBuilder.build(["SA_" + '|'.join(params)], last_features, last_channels)
                layers.append(SelfAttentionEncoder.from_layer(layer, int(num_layers)))

            # Flatten layer
            elif layer_type == "F":
                layers.append(Flatten())
                last_features = np.product(last_features).astype(int) * last_channels
                last_channels = 1

            # Transpose layer (dimension-pairwise)
            elif layer_type == "T":
                dim0, dim1 = map(int, layer_params.split('|'))
                layers.append(Transpose(dim0, dim1))

                # TODO: fix that in a more robust way - like storing the feature/channel dimension index
                # last_features, last_channels = last_channels, last_features

            # MaxPool1d layer
            elif layer_type == "M1":
                param_dict = ModelBuilder.parse_params(layer_params.split('|'), ModelBuilder.conv_param_keys, int)
                layers.append(nn.MaxPool1d(**param_dict))
                last_features = floor(last_features / param_dict.get("kernel_size"))

            # MaxPool2d layer
            elif layer_type == "M2":
                param_dict = ModelBuilder.parse_params(layer_params.split('|'), ModelBuilder.conv_param_keys, eval)
                layers.append(nn.MaxPool2d(**param_dict))
                last_features = np.floor(last_features / np.array(param_dict.get("kernel_size")))

            # PointNet-style aggregation - MaxPool1d layer over last dimension
            elif layer_type == "MaxLastDim":
                layers.append(MaxLastDim())
                last_features = 1

            # PointNet-style aggregation - MaxPool1d layer with concatenation of result to source tensor
            elif layer_type == "ConcatWithMaxedLastDim":
                layers.append(ConcatWithMaxedLastDim())
                last_channels *= 2

            # BatchNorm1d layer
            elif layer_type == "BN1":
                layers.append(nn.BatchNorm1d(last_channels))

            else:
                raise ValueError("dont know %s module" % layer_type)

        return MaskableSequential(*layers), last_channels, last_features


class EnvUtils:
    @staticmethod
    def from_env_state(env_state: GymTuple) -> tuple:
        """
        Convert state from environment state to this model state
        :param env_state:
        :return:
        """
        host_state, actors_state = env_state
        host_flat = host_state.view(host_state.size(0), -1)
        actors_flat = actors_state.view(actors_state.size(0), -1)
        return host_flat, actors_flat

    @staticmethod
    def mask_logits(action_mask: np.array):
        """
        Turns all masked logits to -inf (actually to torch.min_value for stability)
        :param action_mask: an np.array containing ones or zeros for allowed or restricted actions, respectively
        :return:
        """
        # Clamping to minimum float32 value because -inf causes negative probs
        return torch.clamp(torch.log(action_mask.float()), min=np.finfo(np.float32).min)


class ModelInits:
    @staticmethod
    def build_xavier_relu(gain: str = 'relu', logging: bool = False):
        def init(m):
            if hasattr(m, 'weight'):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if logging:
                    logger.info(
                        'ModelInits initialized weights for layer %s with init_xavier_relu(gain=%s)' % (str(m), gain))
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.)
                if logging:
                    logger.info('ModelInits initialized bias for layer %s with 0 constant' % str(m))

        return init

    @staticmethod
    def build_normal(mean: float = 0., std: float = .1, bias: float = 0., logging: bool = False):
        def init(m):
            if hasattr(m, 'weight'):
                nn.init.normal_(m.weight, mean=mean, std=std)
                if logging:
                    logger.info('ModelInits initialized weights for layer %s with normal(mean=%s, std=%s)' % (
                        str(m), mean, std))
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, bias)
                if logging:
                    logger.info('ModelInits initialized bias for layer %s with 0 constant' % str(m))

        return init
