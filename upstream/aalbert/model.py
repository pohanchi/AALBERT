import torch
import torch.nn as nn
import math
from src.module import transformer

class AALBERT(nn.Module):
    def __init__(self, config):
        super().__init__()

        config['common']['share_across_layer'] = True

        self.extractor = transformer.PretrainedModel(config)

        spec_head_config = {**config['common'], **config['transform'] }

    def forward(self, spec_input, pos_idx, att_mask, layer_index=None, mode=None):
        sequence_output, all_attentions = self.extractor(spec_input, pos_idx, att_mask, mode)

        if layer_index is None:
            return sequence_output, None, all_attentions
        else:
            return sequence_output[layer_index], None, all_attentions[layer_index]

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

class SpecHead(nn.Module):
    def __init__(self, hidden_dim, act_fn, downsample_rate, output_dim, norm_eps, name, **kwargs):
        super(SpecHead, self).__init__()
        self.name = name + "_head"
        self.output_dim = output_dim
        self.downsample_rate = downsample_rate
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.transform_act_fn = ACT2FN[act_fn]
        self.LayerNorm = torch.nn.LayerNorm(hidden_dim, eps=norm_eps)
        self.output = nn.Linear(hidden_dim, output_dim * downsample_rate)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        linear_output = self.output(hidden_states)
        return linear_output, hidden_states