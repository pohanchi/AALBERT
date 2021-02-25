import os
import numpy as np
import torch
import argparse
import yaml


def pretrain_args():
    parser = argparse.ArgumentParser()

    # train or test for this experiment
    parser.add_argument('-m', '--mode', choices=['train', 'evaluate'])

    # use a ckpt as the experiment initialization
    # if set, all the following args and config will be overwrited by the ckpt, except args.mode
    parser.add_argument('-e', '--past_exp', metavar='{CKPT_PATH,CKPT_DIR}', help='Resume training from a checkpoint for evaluate it')

    parser.add_argument('-c', '--config', help='The yaml file for configuring the whole experiment except the upstream model')
    parser.add_argument('-g', '--model_config', help='The config file for constructing the pretrained model')
    
    # experiment directory, choose one to specify
    # expname uses the default root directory: result/downstream
    parser.add_argument('-n', '--expname', help='Save experiment at result/downstream/expname')
    parser.add_argument('-p', '--expdir', help='Save experiment at expdir')

    # options
    parser.add_argument('--seed', default=1337, type=int)
    parser.add_argument('--device', default='cuda', help='model.to(device)')

    args = parser.parse_args()

    if args.expdir is None:
        args.expdir = f'result/pretrain/{args.expname}'

    if os.path.isfile(f'{args.expdir}/{args.mode}_finished'):
        exit(0)

    if args.past_exp:
        # determine checkpoint path
        if os.path.isdir(args.past_exp):
            ckpt_pths = glob.glob(f'{args.past_exp}/states-*.ckpt')
            assert len(ckpt_pths) > 0
            ckpt_pths = sorted(ckpt_pths, key=lambda pth: int(pth.split('-')[-1].split('.')[0]))
            ckpt_pth = ckpt_pths[-1]
        else:
            ckpt_pth = args.past_exp

        print(f'[Message] - Resume from {ckpt_pth}')


        # load checkpoint
        ckpt = torch.load(ckpt_pth, map_location='cpu')

        def update_args(old, new):
            old_dict = vars(old)
            new_dict = vars(new)
            old_dict.update(new_dict)
            return Namespace(**old_dict)

        # overwrite args and config
        mode = args.mode
        args = update_args(args, ckpt['Args'])
        config = ckpt['Config']
        args.mode = mode
        args.past_exp = ckpt_pth
    
    else:

        print('[Message] - Start a new experiment')
        os.makedirs(args.expdir, exist_ok=True)

        if args.config is None:
            args.config = f'./upstream/{args.upstream}/pretrain/config/pretrain_config.yaml'
        with open(args.config, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        copyfile(args.config, f'{args.expdir}/pretrain_config.yaml')

        if args.model_config is None:
            args.model_config = f'./upstream/{args.upstream}/pretrain/config/model_config.yaml'
        with open(args.model_config, 'r') as file:
            model_config = yaml.load(file, Loader=yaml.FullLoader)
        
        copyfile(args.model_config, f'{args.expdir}/config/model_config.yaml')

    return args, config, model_config

def set_fix_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def main():
    # get config and arguments
    args, config, model_config = pretrain_args()

    set_fix_seed(args)


if __name__ == "__main__":
    main()