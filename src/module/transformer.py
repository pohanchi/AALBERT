import torch
import torch.nn as nn

def cal_angle(position, hid_idx,hidden_dim):
    return position / np.power(10000, 2 * (hid_idx // 2) / hidden_dim)

def get_posi_angle_vec(position,hidden_dim):
    return [cal_angle(position, hid_j,hidden_dim) for hid_j in range(hidden_dim)]

@lru_cache(maxsize=1)
def static_position_table_f(hidden_dim,max_length=3000):

    sinusoid_table          = np.array([get_posi_angle_vec(pos_i,hidden_dim) for pos_i in range(3000)])
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
    idx = idxes % position_table.weights.shape[0]
    return position_table(idx)

def nontrainable_position_enc(position_table, idxes):
    max_idx = max(idxes)
    if max_idx >= position_table.shape[0]:
        sinusoid_table = static_position_table_f(hidden_dim, max_timestep).to(idxes.device)
    return sinusoid_table(idxes)

def positional_fn(trainable):
    if trainable:
        return trainable_position_enc
    else:
        # sinusoid table
        return nontrainable_position_enc

class InputRep(nn.Module):

    def __init__(self, config, input_dim):
        super(InputRep, self).__init__()
        
        # common setting
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.spec_transform = nn.Linear(input_dim * config.downsample_rate, config.hidden_dim)
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_dim, eps=config.norm_eps)
        self.dropout = nn.Dropout(config.dropout)

        # position setting
        self.position_embed = positional_table_gen(config.trainable, self.hidden_dim, config.max_timestep)
        self.position_fn = position_fn(config.trainable)
    
    def forward(self, spec, pos_idx):
        spec_transformed = self.spec_transform(spec)
        pos_embed = self.position_fn(self.position_embed, pos_idx)
        out_embed = spec_transformed + pos_embed
        norm_embed = self.LayerNorm(out_embed)
        norm_embed = self.dropout(norm_embed)

        return norm_embed

class SelfAttention(nn.Module):
    def __init__(self, config):
        super(SelfAttention, self).__init__()
        if config.hidden_size % config.attention_head != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.attention_head))
        
        self.attention_head = config.attention_head
        self.attention_head_size = int(config.hidden_dim / config.attention_head)
        self.all_head_size = self.attention_head * self.attention_head_size

        self.query = nn.Linear(config.hidden_dim, self.all_head_size)
        self.key = nn.Linear(config.hidden_dim, self.all_head_size)
        self.value = nn.Linear(config.hidden_dim, self.all_head_size)

        self.dropout = nn.Dropout(config.dropout)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[
            :-1] + (self.attention_head, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, head_mask=None):
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

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer, attention_probs

class SelfOutput(nn.Module):
    def __init__(self, config):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.LayerNorm = MockingjayLayerNorm(
            config.hidden_dim, eps=config.norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.self = SelfAttention(config)
        self.output = SelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask, head_mask)
        self_output, attentions  = self_output
        attention_output = self.output(self_output, input_tensor)
        return attention_output ,attentions

class Intermediate(nn.Module):
    def __init__(self, config):
        super(Intermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_dim, config.intermediate_dim)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class Output(nn.Module):
    def __init__(self, config):
        super(Output, self).__init__()
        self.dense = nn.Linear(config.intermediate_dim, config.hidden_dim)
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_dim, eps=config.norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states