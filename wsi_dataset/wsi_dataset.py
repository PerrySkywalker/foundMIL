from torch.utils.data import Dataset
import os
import torch
import pandas as pd
class WsiDataset(Dataset):
    def __init__(self, data_dir, csv_path, state):
        super().__init__()
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.state = state
        self.slide_data = pd.read_csv(csv_path)
        if state == 'train':
            self.data = self.slide_data.loc[:, 'train'].dropna()
            self.label = self.slide_data.loc[:, 'train_label'].dropna()
        if state == 'val':
            self.data = self.slide_data.loc[:, 'val'].dropna()
            self.label = self.slide_data.loc[:, 'val_label'].dropna()
        if state == 'test':
            self.data = self.slide_data.loc[:, 'test'].dropna()
            self.label = self.slide_data.loc[:, 'test_label'].dropna()
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        slide_id = f'{self.data[idx]}.pt'
        label = int(self.label[idx])
        full_path = os.path.join(self.data_dir, slide_id)
        features = torch.load(full_path)
        return features, label