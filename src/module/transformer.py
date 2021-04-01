import torch
import torch.nn as nn
import functools
from functools import lru_cache
import numpy as np
import copy
import math

def cal_angle(position, hid_idx,hidden_dim):
    return position / np.power(10000, 2 * (hid_idx // 2) / hidden_dim)

def get_posi_angle_vec(position,hidden_dim):
    return [cal_angle(position, hid_j,hidden_dim) for hid_j in range(hidden_dim)]

@lru_cache(maxsize=1)
def static_position_table_f(hidden_dim,max_length=3000):

    sinusoid_table          = np.array([get_posi_angle_vec(pos_i,hidden_dim) for pos_i in range(max_length)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  
    sinusoid_table          = torch.FloatTensor(sinusoid_table).to(dtype=torch.float32)
    
    return nn.Embedding.from_pretrained(sinusoid_table)

def positional_table_gen(trainable, hidden_dim, max_timestep):
    if trainable:
        return nn.Embedding(max_timestep, hidden_dim)
    else:
        # sinusoid table
        return static_position_table_f(hidden_dim, max_timestep)

def trainable_position_enc(position_table, idxes):
    # recycle usage handling long range input
    idx = idxes % position_table.weight.data.shape[0]
    return position_table(idx)

def nontrainable_position_enc(position_table, idxes):
    max_idx = idxes.max()
    if max_idx >= position_table.weight.data.shape[0]:
        sinusoid_table = static_position_table_f(position_table.weight.data.shape[1], max_idx+1).to(position_table.weight.device)
        return sinusoid_table(idxes)
    return position_table(idxes)

def position_fn(trainable):
    if trainable:
        return trainable_position_enc
    else:
        # sinusoid table
        return nontrainable_position_enc

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

class InputRep(nn.Module):

    def __init__(self, input_dim, hidden_dim, downsample_rate, norm_eps, dropout, max_timestep, trainable, **kwargs):
        super(InputRep, self).__init__()
        
        # common setting
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.spec_transform = nn.Linear(input_dim * downsample_rate, hidden_dim)
        self.LayerNorm = torch.nn.LayerNorm(hidden_dim, eps=norm_eps)
        self.dropout = nn.Dropout(dropout)

        # position setting
        self.position_embed = positional_table_gen(trainable, hidden_dim, max_timestep)
        self.position_fn = position_fn(trainable)
    
    def forward(self, spec, pos_idx, mode=None):
        if mode is None:
            spec_transformed = self.spec_transform(spec)
        else:
            spec_transformed = spec

        pos_embed = self.position_fn(self.position_embed, pos_idx)

        out_embed = spec_transformed + pos_embed
        norm_embed = self.LayerNorm(out_embed)
        norm_embed = self.dropout(norm_embed)

        return norm_embed

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, attention_head, dropout, **kwargs):
        super(SelfAttention, self).__init__()
        if hidden_dim % attention_head != 0:
            raise ValueError(
                "The hidden dimension (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_dim, attention_head))
        
        self.attention_head = attention_head
        self.attention_head_size = int(hidden_dim / attention_head)
        self.all_head_size = self.attention_head * self.attention_head_size

        self.query = nn.Linear(hidden_dim, self.all_head_size)
        self.key = nn.Linear(hidden_dim, self.all_head_size)
        self.value = nn.Linear(hidden_dim, self.all_head_size)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[
            :-1] + (self.attention_head, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / \
            math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in MockingjayModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer, attention_probs

class SelfOutput(nn.Module):
    def __init__(self, hidden_dim, norm_eps, dropout, **kwargs):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.LayerNorm = torch.nn.LayerNorm(
            hidden_dim, eps=norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.self = SelfAttention(**config)
        self.output = SelfOutput(**config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        self_output, attentions  = self_output
        attention_output = self.output(self_output, input_tensor)
        return attention_output ,attentions

class Intermediate(nn.Module):
    def __init__(self, hidden_dim, act_fn, intermediate_dim, **kwargs):
        super(Intermediate, self).__init__()
        self.dense = nn.Linear(hidden_dim, intermediate_dim)
        if isinstance(act_fn, str) or (sys.version_info[0] == 2 and isinstance(act_fn, unicode)):
            self.intermediate_act_fn = ACT2FN[act_fn]
        else:
            self.intermediate_act_fn = act_fn

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class Output(nn.Module):
    def __init__(self, intermediate_dim, hidden_dim, norm_eps, dropout, **kwargs):
        super(Output, self).__init__()
        self.dense = nn.Linear(intermediate_dim, hidden_dim)
        self.LayerNorm = torch.nn.LayerNorm(hidden_dim, eps=norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class Layer(nn.Module):
    def __init__(self, config):
        super(Layer, self).__init__()
        self.attention = Attention(config)
        self.intermediate = Intermediate(**config)
        self.output = Output(**config)

    def forward(self, hidden_states, attention_mask):
        attention_output, attentions = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        return layer_output, attentions

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        layer = Layer(config)
        if config['share_across_layer']:
            # shallow copy
            self.layer = nn.ModuleList([layer for _ in range(config['num_layers'])])
        else:
            # deep copy
            self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config['num_layers'])])

    def forward(self, hidden_states, attention_mask, output_attention=False,all_hidden=False):
        all_encoder_layers = []
        all_attentions = []
        for i, layer_module in enumerate(self.layer):
            hidden_states, attentions = layer_module(
                hidden_states, attention_mask)
            
            if output_attention:
                all_attentions.append(attentions.detach().cpu())
            
            if all_hidden:
                all_encoder_layers.append(hidden_states)

        if not all_hidden:
            all_encoder_layers.append(hidden_states)
        
        if not output_attention:
            all_attentions.append(None)

        return all_encoder_layers, all_attentions


class InitModel(nn.Module):
    
    """ 
    An abstract class to handle weights initialization.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(InitModel, self).__init__()
        self.config = config

    def init_weights(self, module):
        """ Initialize the weights. """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            # necessary for static position embedding
            if module.weight.requires_grad:
                module.weight.data.normal_(mean=0.0, std=self.config['common']['init_range'])
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

        


class PretrainedModel(InitModel):
    """
    PretrainedModel 
    """

    def __init__(self, config):
        super(PretrainedModel, self).__init__(config)

        input_reps_config = {**config['common'], **config["transform"], **config['position_embedding']}

        self.input_reps = InputRep(**input_reps_config)

        encoder_reps_config = {**config['common'], **config['attention'], **config["fully_connected"]}

        self.encoder = Encoder(encoder_reps_config)
        self.apply(self.init_weights)

    def forward(self, spec_input, pos_enc, attention_mask, mode, output_attention=False, all_hidden=False):
        if attention_mask is None:
            attention_mask = torch.ones_like(spec_input)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask[:,None,None]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        input_representations = self.input_reps(spec_input, pos_enc, mode)
        encoded_layers, all_attentions = self.encoder(input_representations, extended_attention_mask, output_attention, all_hidden)
        
        return encoded_layers, all_attentions
