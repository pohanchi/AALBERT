import os
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

class PretrainedSystem(pl.LightningModule):

    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return

    def training_step(self, batch, batch_idx):
        return
    
    def configure_optimizers(self):
        return