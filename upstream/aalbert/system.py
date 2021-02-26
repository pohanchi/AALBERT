import os
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from model import AALBERT, SpecHead

class PretrainedSystem(pl.LightningModule):

    def __init__(self, model_config, training_config):
        super().__init__(model_config, training_config)


        self.tradition_feat_extractor = 

        self.pretrained_model = AALBERT(model_config['model'])
        
        self.pretrained_heads = []
        for feat_type in training_config['datarc']['target']:
            if feat_type == "mel":
                output_dim = training_config[feature_type]['n_mels'] * (training_config[feature_type]['input']['delta'] +1)
            elif feat_type == "linear":
                output_dim = training_config[feature_type]['n_freq'] * (training_config[feature_type]['input']['delta'] +1)                
            spechead_config = {**model_config['model']['common'], 
            **model_config['model']['transform'], "output_dim": output_dim, "name": feat_type}
            
            self.pretrained_heads.append(SpecHead(**spechead_config))

        

    def forward(self, x):
        return

    def training_step(self, batch, batch_idx):
        return
    
    def configure_optimizers(self):
        return