#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 18:00:18 2024

@author: homai
"""

import os
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageFile, ImageOps

import torch.utils.data as data
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader

from torch.utils.data import ConcatDataset
import dataloaders as loaders
from fundusloader import selfcensor_loader, fundus_loader

class CombinedDataset(Dataset):
    def __init__(self, fundus_dataset, cifar10_dataset):
        self.fundus_dataset = fundus_dataset
        self.cifar10_dataset = cifar10_dataset

    def __len__(self):
        return max(len(self.fundus_dataset), len(self.cifar10_dataset))

    def __getitem__(self, idx):
        if idx < len(self.fundus_dataset):
            return self.fundus_dataset[idx]
        else:
            return self.cifar10_dataset[idx - len(self.fundus_dataset)]

from torch.utils.data import DataLoader, ConcatDataset

def combined_loader(fundus_loader, cifar_loader, batch_size, shuffle, num_workers):
    combined_dataset = ConcatDataset([fundus_loader.dataset, cifar_loader.dataset])
    combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return combined_loader



# Usage example:
fundus_trnf = transforms.Compose([transforms.ToTensor()])
naturl_trnf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


path      = '../results/metamodel_for_24-03-04_09-33-16/Meta_24-03-04_09-33-16__24-03-11_21-29-16'
ORIGA     = pd.read_csv(os.path.join(path, 'ORIGA_probs.csv')); 

fundus_loader  = fundus_loader(224, ORIGA, fundus_trnf, 32, False, False, 0)
natural_loader = loaders.SVHN(naturl_trnf, batch_size=100, shuffle=False, num_workers=0)


combined_loader = combined_loader(fundus_loader, natural_loader, batch_size=32, shuffle=True, num_workers=4)

print(combined_loader)

for idx, an in enumerate(combined_loader):
    print(idx)
        
    # if batch_idx == 0:
    #     plt.close(); plt.imshow(xs[0].permute(1, 2, 0)); plt.tight_layout(); plt.savefig(f'../results/eval_{data_name}.png') #  plt.imshow(xs[0,0,...]);


