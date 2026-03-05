import os
import pandas as pd
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(self, data_dir='', ref=False, lr_train=False):
        self.ref = ref
        self.lr_train = lr_train
        if self.lr_train:
          self.ref = 'convo'
        
        if self.ref == 'convo':
             self.audio_dir = "./data/train"
        elif data_dir:
             self.audio_dir = f"./data/{data_dir}"
        else:
             self.audio_dir = "./data/test"

        tmp = os.listdir(self.audio_dir)
        self.file_names = [f for f in tmp if f.endswith(".wav")]
        
        if self.ref: # transcription only
            if not self.lr_train:
                self.audio_label = pd.read_excel(f"{self.audio_dir}/transcription_swe.xlsx")
                self.audio_label.set_index('file_name', inplace=True)
                self.ref_path = f"{self.audio_dir}/transcription_swe.xlsx"

            else:
                self.audio_label = pd.read_excel(f"{self.audio_dir}/train_swe.xlsx") # speaker id + segment + transcript
                self.ref_path = f"{self.audio_dir}/train_swe.xlsx"


    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.file_names[idx])
        if self.ref:
             transcript = self.audio_label.loc[self.file_names[idx]]['transcript']
             return audio_path, transcript
        else:
             return audio_path

    def get_ref_path(self):
        if not self.ref:
            raise Exception()
        return self.ref_path
