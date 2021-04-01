import os
import torch
import torchaudio
import librosa
import numpy as np 
import pathlib
import pytorch_lightning
import pickle
import tqdm
import random
import time
from torch.utils.data import Dataset, DataLoader
from librosa.util import find_files
from torchaudio.sox_effects import apply_effects_file
from pytorch_lightning import LightningDataModule
from pathlib import Path


EFFECTS = [
["channels", "1"],
["rate", "16000"],
["gain", "-3.0"],
["silence", "1", "0.1", "0.1%", "-1", "0.1", "0.1%"],
]

class DownstreamDataset(Dataset):
    def __init__(self, data_config, max_timestep=None):
        super().__init__()

        self.datarc = data_config
        self.root = self.datarc['file_path']
        self.root_key = list(self.datarc['file_path'].keys())
        self.max_timestep = max_timestep
        self.all_speakers = []
        self.data_list = []
        self.prepare_data()
    
    def prepare_data(self):

        for index in range(len(self.root_key)):
            
            cache_path = f'./downstream/libri_360_speaker/path_cache/{self.root_key[index]}.p' 
            p = Path(self.root[self.root_key[index]])

            if os.path.isfile(cache_path):
                cache_wavs_dict = pickle.load(open(cache_path,"rb"))
                self.all_speakers.extend(list(cache_wavs_dict.keys()))
                for speaker_id in list(cache_wavs_dict.keys()):
                    for wavs in cache_wavs_dict[speaker_id]:
                        data_type = wavs.split(".")[-1]
                        utterance_id = "/".join(str(p/speaker_id/wavs).split("/")[-3:]).replace(f".{data_type}","").replace("/","-")
                        self.data_list.append([str(p / speaker_id / wavs), utterance_id])

            else:
                speaker_wav_dict = {}
                print("search all wavs paths")
                start = time.time()

                speaker_dirs = [f.path.split("/")[-1] for f in os.scandir(self.root[self.root_key[index]]) if f.is_dir()]

                for speaker in tqdm.tqdm(speaker_dirs):
                    speaker_dir = p / speaker
                    wav_list = find_files(speaker_dir)
                    speaker_wav_dict[speaker] = []
                    for wav in wav_list:
                        data_type = wav.split(".")[-1]
                        utterance_id = "/".join(str(speaker_dir/wav).split("/")[-3:]).replace(f".{data_type}","").replace("/","-")
                        self.data_list.append([str(speaker_dir/wav), utterance_id])
                        speaker_wav_dict[speaker].append("/".join(wav.split("/")[-2:]))
                end = time.time() 
                print(f"search all wavs paths costs {end-start} seconds")
                print(f"save wav paths to {cache_path}! so we can directly load all_path in next time!")
                pickle.dump(speaker_wav_dict, open(cache_path,"wb"))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        wav, _ = apply_effects_file(str(self.data_list[idx][0]), EFFECTS)
        wav = wav.squeeze(0)
        length = wav.shape[0]

        if self.max_timestep is not None:
            if length > self.max_timestep:
                start = random.randint(0, int(length-self.max_timestep))
                wav = wav[start:start+self.max_timestep]
                length = self.max_timestep
  
        return wav, self.data_list[idx][1]  
    
    def collate_fn(self, samples):
        
        wavs, labels, = [], []

        for wav,label in samples:
            wavs.append(wav)
            labels.append(label)
        return wavs, labels


class DownstreamDataModule(LightningDataModule):
    def __init__(self, data_config, max_timestep=None):
        super().__init__()

        self.datarc = data_config
        self.root = self.datarc['file_path']
        self.root_key = list(self.datarc['file_path'].keys())
        self.max_timestep = max_timestep
        self.dataloader_config = self.datarc['dataloader']
    
    def prepare_data(self):
        dataset_config = {"data_config": self.datarc,"max_timestep": self.max_timestep}
        PretrainDataset(**dataset_config)

    def setup(self, stage=None):        
        if stage == "fit" or stage is None:
            dataset_config = {"data_config": self.datarc,"max_timestep": self.max_timestep}
            self.dataset_full = DownstreamDataset(**dataset_config)
            self.dataset_train = self.dataset_full

    def train_dataloader(self):

        return DataLoader(self.dataset_train, **self.dataloader_config, collate_fn=self.dataset_train.collate_fn)
    
    def val_dataloader(self):
        
        return

    def test_dataloader(self):
        
        return