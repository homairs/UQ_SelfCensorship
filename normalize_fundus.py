#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 12:30:34 2023

@author: homai
"""

import os
import time
import pandas as pd
from PIL import Image

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from fundusloader import fundus_loader


def get_mean_sd_to_normalize_fundus(data_file_path, img_siz):
    # Define a transform for Fundus data
    transform_fundus = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    dataloader = fundus_loader(img_siz, data_file_path, transform_fundus, 100, False, False, 0)
    
    mean = torch.zeros(3)
    sd = torch.zeros(3)
    total_samples = 0
    
    start_time = time.time()
    for batch in dataloader:
        data, _, _, _ = batch
        
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        
        mean += data.mean(2).sum(0)
        sd   += data.std(2).sum(0)
        total_samples += batch_samples
    
    mean /= total_samples
    sd /= total_samples
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f'Time taken: {elapsed_time/60} minutes')
    # print(f'Mean of fundus data= {mean}, and SD= {sd}')
    return mean, sd

