import torch
import torch.nn as nn
from src.module import transformer



class AALBERT(nn.Module):
    def __init__(self, config):

        config['common']['share_across_layer'] = True

        self.extractor = transformer.PretrainedModel(config)

        spec_head_config = {**config['common'], **config['transform'] }

    def forward(self, spec_input, pos_idx, attention_mask, layer_index=None, mode=None):
        sequence_output, all_attentions = self.extractor(spec_input, pos_idx, attention_mask, mode)

        if layer_index is None
            return sequence_output, None, all_attentions
        else:
            return sequence_output[layer_index], None, all_attentions[layer_index]
    