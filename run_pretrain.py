import os
import numpy as np
import torch
import argparse
import yaml
import random
import importlib
from shutil import copyfile
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from utils import ProgressBar

def pretrain_args():
    parser = argparse.ArgumentParser()

    # use a ckpt as the experiment initialization
    # if set, all the following args and config will be overwrited by the ckpt, except args.mode
    parser.add_argument('-e', '--past_exp', metavar='{CKPT_PATH,CKPT_DIR}', help='Resume training from a checkpoint for evaluate it')

    parser.add_argument('-c', '--config', help='The yaml file for configuring the whole experiment except the upstream model')
    parser.add_argument('-g', '--model_config', help='The config file for constructing the pretrained model')
    parser.add_argument('-u', '--upstream', choices=os.listdir('upstream/'))

    # experiment directory, choose one to specify
    # expname uses the default root directory: result/downstream
    parser.add_argument('-n', '--expname', help='Save experiment at result/pretrain/expname')
    parser.add_argument('-p', '--expdir', help='Save experiment at expdir')

    # options
    parser.add_argument('--seed', default=1337, type=int)
    parser.add_argument('--device', default='cuda', help='model.to(device)')

    args = parser.parse_args()

    if args.expdir is None:
        args.expdir = f'result/pretrain/{args.expname}'


    print('[Message] - Start a new experiment')
    if args.expdir is None:
        args.expdir = f'result/pretrain/{args.expname}'
    os.makedirs(args.expdir, exist_ok=True)

    if args.config is None:
        args.config = f'./upstream/{args.upstream}/pretrain_config.yaml'
    with open(args.config, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    copyfile(args.config, f'{args.expdir}/pretrain_config.yaml')

    if args.model_config is None:
        args.model_config = f'./upstream/{args.upstream}/model_config.yaml'
    with open(args.model_config, 'r') as file:
        model_config = yaml.load(file, Loader=yaml.FullLoader)
        
    copyfile(args.model_config, f'{args.expdir}/model_config.yaml')

    return args, config, model_config

def set_fixed_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(args.seed)

def main():
    # get config and arguments
    args, config, model_config = pretrain_args()
    
    root_name = Path("./upstream") / args.upstream
    with open(root_name / config['input']['config_path'], 'r') as file:
        input_config = yaml.load(file, Loader=yaml.FullLoader)
    copyfile(input_config, f'{args.expdir}/input_config.yaml')

    for target in config['target']:
        with open(root / target, 'r') as file:
            target_config = yaml.load(file, Loader=yaml.FullLoader)
        copyfile(target_config, f'{args.expdir}/target_config_{target['feature_type']}.yaml')

    set_fixed_seed(args)

    system_config = {"args":args, "training_config": config, "model_config":model_config}

    module_path = f'upstream.{args.upstream}'
    system = importlib.import_module(module_path +'.system')
    dataset = importlib.import_module(module_path+ '.dataset')

    pretrained_system = system.PretrainedSystem(**system_config)
    datamodule_config = {"data_config": config['datarc'], "max_timestep": config['datarc']['max_timestep']}
    prerained_dataset = dataset.PretrainedDataModule(**datamodule_config)
    wandb_logger = WandbLogger(name=args.expname,save_dir=args.expdir,config=system_config)
    checkpoint_callback = ModelCheckpoint(monitor="train_loss", save_top_k=3, mode="min")
    program_callback = ProgressBar()
    trainer_config = { **config['trainer_config'], 'default_root_dir':  args.expdir, "logger": wandb_logger, "weights_save_path": args.expdir,
     "callbacks": [checkpoint_callback, program_callback]}

    trainer = Trainer(**trainer_config)
    trainer.fit(pretrained_system, prerained_dataset)


if __name__ == "__main__":
    main()