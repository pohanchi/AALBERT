import torch
import importlib
import yaml
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.nn.utils.rnn import pad_sequence
from schedulers import *
from .module import Model

class DownstreamSystem(pl.LightningModule):
    def __init__(self, args, training_config, **kwargs):
        super(DownstreamSystem, self).__init__()
        self.args = args
        self.training_config = training_config
        self.accumulated_step = training_config['trainer_config']['accumulate_grad_batches']
        self.logging_step = training_config['trainer_config']['log_every_n_steps']
        self.optimizer_config = training_config['optimizer']
        self.modelrc = training_config['modelrc']
        
        if training_config.get("scheduler", None):
            self.scheduler_config = training_config['scheduler']
        else:
            self.scheduler_config = None

        self.upstream = self.get_upstream(args)
        self.downstream = self.get_downstream(self.modelrc)
        self.objective_loss = eval(f"nn.{self.modelrc['loss_function']}")()
        self.save_hyperparameters()
    
    def get_upstream(self, args):
        with open(args.model_config,"r") as f:
           model_config = yaml.safe_load(f)
        with open(args.upstream_training_config, "r") as f:
           upstream_config = yaml.safe_load(f)

        upstream_config['datarc']['input']['config_path'] = "/".join(args.upstream_training_config.split("/")[:-1]) + "/" + "input_config.yaml"
        upstream_config['datarc']['target'][0]['config_path'] = "/".join(args.upstream_training_config.split("/")[:-1]) + "/" + "target_config.yaml"

        module_path = f'upstream.{args.upstream}'
        system = importlib.import_module(module_path +'.system')
        PretrainedSystem = system.PretrainedSystem.load_from_checkpoint(args.ckpt, args={}, model_config=model_config, training_config=upstream_config)
        return PretrainedSystem
    
    def get_downstream(self, modelrc):
        input_dim = self.upstream.get_output_dim()
        modelrc['input_dim'] = input_dim
        modelrc.update(modelrc[modelrc['model_select']])
        return eval(modelrc['model_select'])(**modelrc)
        


    def forward(self, batch):
        wav, label = batch
        rep, feat_input ,_ = self.upstream.forward(wav)
        length = [len(feat) for feat in feat_input]
        predict = self.downstream(rep, length)
        return predict



    def training_step(self, batch, batch_idx):
        wav, label = batch
        rep, feat_input ,_ = self.upstream.forward(wav)
        length = [len(feat) for feat in feat_input]
        predict = self.downstream(rep, length)
        loss = self.objective_loss(predict,torch.LongTensor(label).to(self.device))
        if (batch_idx+1) % self.accumulated_step == 0:
            if (self.global_step+1) % self.logging_step == 0:
                self.log("train_loss", loss, on_step=True)
                self.print(f"global_step={self.global_step+1}, loss = {loss}")

        return loss
    
    def validation_step(self, batch, batch_idx):
        wav, label = batch
        rep, feat_input ,_ = self.upstream.forward(wav)
        length = [len(feat) for feat in feat_input]
        predict = self.downstream(rep, length)

        predicted_classid = predict.max(dim=-1).indices
        loss = self.objective_loss(predict,torch.LongTensor(label).to(predict.device))

        return {'val_loss': loss, 'val_pred': predicted_classid, "val_label": torch.LongTensor(label).to(predict.device)}
    
    def validation_epoch_end(self, validation_step_outputs):
        predicts = []
        labels = []
        total_loss = 0
        for out in validation_step_outputs:
           predicts.extend(out['val_pred'])
           labels.extend(out['val_label'])
           total_loss += out['val_loss'].item()
        total_loss /= len(validation_step_outputs)
        predicts = torch.IntTensor(predicts)
        labels = torch.IntTensor(labels)
        accuracy = torch.sum((predicts == labels).view(-1)) / len(predicts)
        
        metrics = {'val_acc': accuracy, 'val_total_loss': total_loss}
        self.log('val_acc', accuracy)
        self.log('val_total_loss', total_loss)
        self.log_dict(metrics)
        self.print(f'Validation Accuracy is {accuracy:.2f} ..')
        self.print(f'Validation Loss is {total_loss:.5f} ..')
    
    def test_step(self, batch, batch_idx):
        wav, label = batch
        rep, feat_input ,_ = self.upstream.forward(wav)
        length = [len(feat) for feat in feat_input]
        predict = self.downstream(rep, length)
        predicted_classid = predict.max(dim=-1).indices
        loss = self.objective_loss(predict,torch.LongTensor(label).to(self.device))
        return {'test_loss': loss, 'test_pred': predicted_classid, "test_label": label}
    
    def test_epoch_end(self, test_step_outputs):
        predicts = []
        labels = []
        total_loss = 0
        for out in validation_step_outputs:
           predicts.extend(out['pred'])
           labels.extend(out['label'])
           total_loss += out['loss'].item()
        total_loss /= len(validation_step_outputs)
        predicts = torch.IntTensor(predicts)
        labels = torch.IntTensor(labels)
        accuracy = torch.sum((predicts == labels).view(-1)) / len(predicts)
        
        metrics = {'test_acc': accuracy, 'test_total_loss':total_loss}
        self.log('test_acc', accuracy)
        self.log('test_total_loss', total_loss)
        self.log_dict(metrics)
        self.print(f'Test Accuracy is {accuracy:.2f} ..')
        self.print(f'Test Loss is {total_loss:.5f} ..')

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def configure_optimizers(self):
    
        no_decay = ["LayerNorm", "bias"]

        downstream_model_decay_parameters = [
            p for n, p in self.downstream.named_parameters() if not any(nd in n for nd in no_decay)]
        downstream_model_nodecay_parameters = [
            p for n, p in self.downstream.named_parameters() if any(nd in n for nd in no_decay)]
        
        all_decay_parameters = downstream_model_decay_parameters
        all_nodecay_parameters = downstream_model_nodecay_parameters
        
        if self.args.finetune:
            pretrained_model_decay_parameters = [
                p for n, p in self.pretrained_model.named_parameters() if not any(nd in n for nd in no_decay)]
            pretrained_model_nodecay_parameters = [
                p for n, p in self.pretrained_model.named_parameters() if any(nd in n for nd in no_decay)]
            all_decay_parameters += pretrained_model_decay_parameters
            all_nodecay_parameters += pretrained_model_nodecay_parameters

        if not self.optimizer_config.get('weight_decay', None):
            self.optimizer_config['weight_decay'] = 0.01

        all_parameters = [
            {
                "params": all_decay_parameters,
                "weight_decay": self.optimizer_config['weight_decay'],
            },
            {
                "params": all_nodecay_parameters,
                "weight_decay": 0.0
            },
        ]

        optimizer_config = copy.deepcopy(self.optimizer_config)
        optimizer = eval(f"torch.optim.{optimizer_config.pop('name')}")(
            params=all_parameters, **optimizer_config)

        if self.scheduler_config:
            scheduler_config = copy.deepcopy(self.scheduler_config)
            scheduler = eval(f"get_{scheduler_config.pop('name')}")(
                optimizer=optimizer, **scheduler_config, num_training_steps=len(self.train_dataloader())*self.training_config['trainer_config']['max_epochs'])
            return [optimizer], [scheduler]
        else:
            return [optimizer]


    
        

