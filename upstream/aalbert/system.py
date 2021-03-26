import os
import torch
import importlib
import torch.nn.functional as F
import pytorch_lightning as pl
import IPython
import pdb
from torch import nn
from shutil import copyfile
from torch.nn.utils.rnn import pad_sequence
from .model import AALBERT, SpecHead
from schedulers import *

examples_wavs = torch.randn(16000)


class PretrainedSystem(pl.LightningModule):

    def __init__(self, args, model_config, training_config, **kwargs):
        super().__init__()

        self.args = args
        self.model_config = model_config
        self.accumulated_step = training_config['trainer_config']['accumulate_grad_batches']
        self.logging_step = training_config['trainer_config']['log_every_n_steps']
        self.downsample_rate = model_config['model']['transform']['downsample_rate']
        tradition_feat = getattr(importlib.import_module(
            'hubconf'), training_config['datarc']['input']['feature_type'])
        self.tradition_feat_extractor = tradition_feat(
            config=training_config['datarc']['input']['config_path'])
        self.masking_strategy = training_config['masking_strategy']
        self.pretrained_model = AALBERT(model_config['model'])
        self.optimizer_config = training_config['optimizer']

        # example wav forward to tradition feature extractor
        _ = self.tradition_feat_extractor([examples_wavs])

        if training_config.get("scheduler", None):
            self.scheduler_config = training_config['scheduler']
        else:
            self.scheduler_config = None

        self.pretrained_heads = nn.ModuleList()
        self.target_feat_extractors = nn.ModuleList()

        index = 0
        for feat in training_config['datarc']['target']:

            tradition_feat = getattr(importlib.import_module(
                "hubconf"), feat['feature_type'])
            tradition_feat_system = tradition_feat(config=feat['config_path'])

            spechead_config = {**model_config['model']['common'],
                               **model_config['model']['transform'], 
                               "output_dim": tradition_feat_system.get_output_dim(), 
                               "name": feat['feature_type']}

            self.target_feat_extractors.append(
                tradition_feat(config=feat['config_path']))
            self.pretrained_heads.append(SpecHead(**spechead_config))

            # example wav forward to tradition feature extractor
            _ = self.target_feat_extractors[index]([examples_wavs])
            index += 1

        self.objective_loss = eval(
            f"nn.{training_config['loss_function']}")(reduction='sum')
        self.save_hyperparameters()

    def downsample(self, feats):

        input_feats = []
        left_feats = []
        for feat in feats:
            length_after_downsample = (
                feat.shape[0] // self.downsample_rate) * self.downsample_rate
            left_feat = feat[:length_after_downsample]
            stack_feat = left_feat.reshape(
                length_after_downsample // self.downsample_rate, -1)
            input_feats.append(stack_feat)
            left_feats.append(left_feat)

        return input_feats, left_feats

    def generate_mask_frame(self, stack_feats):

        mask_labels = []
        for stack in stack_feats:
            mask_label = torch.zeros(stack.size(0))
            valid_box_size = (
                len(stack) // self.masking_strategy['mask_consecutive'])
            remainder = len(stack) % self.masking_strategy['mask_consecutive']
            probs = torch.zeros(valid_box_size).fill_(
                self.masking_strategy['mask_proportion']/self.masking_strategy['mask_consecutive'])

            # generate start point to mask frame
            start_indexes = torch.nonzero(torch.bernoulli(probs))
            offset = random.randint(0, remainder)
            base_indexes = start_indexes.expand(start_indexes.size(
                0), self.masking_strategy['mask_consecutive'])
            consecutive_offset = torch.arange(self.masking_strategy['mask_consecutive']).long(
            ).expand(base_indexes.size(0), self.masking_strategy['mask_consecutive'])
            indexes = (base_indexes + consecutive_offset + offset).reshape(-1)
            stack[indexes] = 0
            mask_label[indexes] = 1
            mask_labels.append(mask_label)

        return stack_feats, mask_labels

    def generate_att_mask(self, feats):

        att_mask = []
        for feat in feats:
            att_mask.append(torch.ones(len(feat)))
        return att_mask

    def generate_pos_idxes(self, att_mask_seq):

        pos_idxes = torch.arange(len(att_mask_seq[0]))
        pos_idxes = pos_idxes.expand(len(att_mask_seq), len(att_mask_seq[0]))

        return pos_idxes

    def forward(self, x, output_attention=False, all_hidden=False, layer_index=None):

        feats = self.tradition_feat_extractor(x)
        stack_feats, _ = self.downsample(feats)
        att_masks = self.generate_att_mask(stack_feats)
        input_feats = pad_sequence(stack_feats, batch_first=True)
        input_att_masks = pad_sequence(
            att_masks, batch_first=True).to(self.device).long()
        input_pos_idxs = self.generate_pos_idxes(
            input_att_masks).to(self.device).long()

        if layer_index is not None:
            all_hidden = True

        forward_config = {"spec_input": input_feats, "att_mask": input_att_masks,
                          "pos_idx": input_pos_idxs, "output_attention": output_attention, "all_hidden": all_hidden}
        all_hidden_states, all_attentions = self.pretrained_model(
            **forward_config)

        if layer_index is not None:
            all_hidden_states = all_hidden_states[layer_index]
        else:
            all_hidden_states = all_hidden_states[-1]
        return all_hidden_states, stack_feats, all_attentions

    def training_step(self, batch, batch_idx):

        wav_list, _ = batch
        with torch.no_grad():
            feats = self.tradition_feat_extractor(wav_list)

        stack_feats, _ = self.downsample(feats)
        att_masks = self.generate_att_mask(stack_feats)
        stack_feats, mask_indexes = self.generate_mask_frame(stack_feats)
        input_feats = pad_sequence(stack_feats, batch_first=True)
        mask_indexes = pad_sequence(mask_indexes, batch_first=True)[
            :, :, None].to(self.device).bool()
        input_att_masks = pad_sequence(
            att_masks, batch_first=True).to(self.device)
        input_pos_idxes = self.generate_pos_idxes(
            input_att_masks).to(self.device)
        forward_config = {"spec_input": input_feats, "att_mask": input_att_masks,
                          "pos_idx": input_pos_idxes, "layer_index": -1}
        last_hidden_states, _, _ = self.pretrained_model(**forward_config)

        loss = 0
        for pred_head_idx in range(len(self.pretrained_heads)):
            wav_list, _ = batch
            with torch.no_grad():
                feats = self.target_feat_extractors[pred_head_idx](wav_list)

            ds_feats, left_feats = self.downsample(feats)
            ds_feats = pad_sequence(ds_feats, batch_first=True)
            reconstruct_feats, _ = self.pretrained_heads[pred_head_idx](
                last_hidden_states)
            masked_spec_loss = self.objective_loss(reconstruct_feats.masked_select(
                mask_indexes), ds_feats.masked_select(mask_indexes)) / torch.sum(mask_indexes) / self.downsample_rate

            loss += masked_spec_loss / len(self.pretrained_heads)

        if (batch_idx+1) % self.accumulated_step == 0:
            if (self.global_step+1) % self.logging_step == 0:
                self.log("train_loss", loss, on_step=True)
                self.print(f"global_step={self.global_step+1}, loss = {loss}")

        return loss

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    # Interface
    def get_output_dim(self):
        return self.model_config['model']['common']['hidden_dim']

    def configure_optimizers(self):

        no_decay = ["LayerNorm", "bias"]

        pretrained_model_decay_parameters = [
            p for n, p in self.pretrained_model.named_parameters() if not any(nd in n for nd in no_decay)]
        pretrained_heads_decay_parameters = [
            p for head in self.pretrained_heads for n, p in head.named_parameters() if not any(nd in n for nd in no_decay)]

        pretrained_model_nodecay_parameters = [
            p for n, p in self.pretrained_model.named_parameters() if any(nd in n for nd in no_decay)]
        pretrained_heads_nodecay_parameters = [
            p for head in self.pretrained_heads for n, p in head.named_parameters() if any(nd in n for nd in no_decay)]

        if not self.optimizer_config.get('weight_decay', None):
            self.optimizer_config['weight_decay'] = 0.01

        all_parameters = [
            {
                "params": pretrained_model_decay_parameters + pretrained_heads_decay_parameters,
                "weight_decay": self.optimizer_config['weight_decay'],
            },
            {
                "params": pretrained_model_nodecay_parameters + pretrained_heads_nodecay_parameters,
                "weight_decay": 0.0
            },
        ]

        optimizer_config = copy.deepcopy(self.optimizer_config)
        optimizer = eval(f"torch.optim.{optimizer_config.pop('name')}")(
            params=all_parameters, **optimizer_config)

        if self.scheduler_config:
            scheduler_config = copy.deepcopy(self.scheduler_config)
            scheduler = eval(f"get_{scheduler_config.pop('name')}")(
                optimizer=optimizer, **scheduler_config)
            return [optimizer], [scheduler]
        else:
            return [optimizer]
