import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np 
from librosa.util import find_files
from torchaudio import load
from torch import nn
import os 
import random
import pickle
import torchaudio
import sys
import time
import glob
import tqdm
import pytorch_lightning
from pathlib import Path
from pytorch_lightning import LightningDataModule


CACHE_PATH = os.path.join(os.path.dirname(__file__), '.cache/')


# Voxceleb 1 Speaker Identification
class SpeakerClassifiDataset(Dataset):
    def __init__(self, mode, file_path, meta_data, max_timestep=None):

        self.root = file_path
        self.speaker_num = 1251
        self.meta_data =meta_data
        self.max_timestep = max_timestep
        self.usage_list = open(self.meta_data, "r").readlines()

        cache_path = os.path.join(CACHE_PATH, f'{mode}.pkl')
        if os.path.isfile(cache_path):
            print(f'Loading file paths from {cache_path}')
            with open(cache_path, 'rb') as cache:
                dataset = pickle.load(cache)
        else:
            dataset = eval("self.{}".format(mode))()
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as cache:
                pickle.dump(dataset, cache)
        print(f'there are {len(dataset)} files found')

        self.dataset = dataset
        self.label = self.build_label(self.dataset)

    # file_path/id0001/asfsafs/xxx.wav
    def build_label(self, train_path_list):

        y = []
        for path in train_path_list:
            id_string = path.split("/")[-3]
            y.append(int(id_string[2:]) - 10001)

        return y
    
    def train(self):

        dataset = []
        print("search specified wav name for training set")
        for string in tqdm.tqdm(self.usage_list):
            pair = string.split()
            index = pair[0]
            if int(index) == 1:
                x = list(self.root.glob("*/wav/" + pair[1]))
                dataset.append(str(x[0]))
        print("finish searching training set wav")
                
        return dataset
        
    def dev(self):

        dataset = []
        print("search specified wav name for dev set")
        for string in tqdm.tqdm(self.usage_list):
            pair = string.split()
            index = pair[0]
            if int(index) == 2:
                x = list(self.root.glob("*/wav/" + pair[1]))
                dataset.append(str(x[0])) 
        print("finish searching dev set wav")

        return dataset       

    def test(self):

        dataset = []
        print("search specified wav name for test set")
        for string in tqdm.tqdm(self.usage_list):
            pair = string.split()
            index = pair[0]
            if int(index) == 3:
                x = list(self.root.glob("*/wav/" + pair[1]))
                dataset.append(str(x[0])) 
        print("finish searching test set wav")

        return dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        wav, sr = torchaudio.load(self.dataset[idx])
        wav = wav.squeeze(0)
        length = wav.shape[0]

        if self.max_timestep !=None:
            if length > self.max_timestep:
                start = random.randint(0, int(length-self.max_timestep))
                wav = wav[start:start+self.max_timestep]
                length = self.max_timestep
  
        return wav, self.label[idx]
        
    def collate_fn(self, samples):        
        wavs, labels = [], []
        for wav, label in samples:
            wavs.append(wav)
            labels.append(label)
        return wavs, labels


class DownstreamDataModule(LightningDataModule):
    def __init__(self, data_config, dataloader_config, **kwargs):
        super().__init__()

        self.datarc = data_config
        self.dataloaderrc = dataloader_config
        root_dir = Path(self.datarc['file_path'])
        self.train_batch_size = self.dataloaderrc.pop("train_batch_size")
        self.eval_batch_size = self.dataloaderrc.pop("eval_batch_size")
        
    def prepare_data(self):
        pass

    def setup(self, stage=None):        
        root_dir = Path(self.datarc['file_path'])

        self.train_dataset = SpeakerClassifiDataset('train', root_dir, self.datarc['meta_data'], self.datarc['max_timestep'])
        self.dev_dataset = SpeakerClassifiDataset('dev', root_dir, self.datarc['meta_data'])
        self.test_dataset = SpeakerClassifiDataset('test', root_dir, self.datarc['meta_data'])

    def train_dataloader(self):

        return DataLoader(self.train_dataset, **self.dataloaderrc, batch_size=self.train_batch_size, collate_fn=self.train_dataset.collate_fn)
    
    def val_dataloader(self):
        
        return DataLoader(self.dev_dataset, **self.dataloaderrc, batch_size=self.eval_batch_size, collate_fn=self.dev_dataset.collate_fn)

    def test_dataloader(self):
        
        return DataLoader(self.test_dataset, **self.dataloaderrc, batch_size=self.eval_batch_size, collate_fn=self.dev_dataset.collate_fn)