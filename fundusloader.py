#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 10:13:48 2023

@author: homai
"""

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image, ImageFile, ImageOps

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, img_siz, file: str, transform = None):
        self.file      = file
        self.img_siz   = img_siz
        self.transform = transform
        
    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):
        label = self.file.loc[idx, 'label']
        img_path = os.path.join(self.file.loc[idx, 'img_dir'], self.file.loc[idx, 'fileName'])   
        # print(img_path)
        
        img = Image.open(img_path).convert('RGB') # .convert('L')
        # print('good?')
        # if self.img_siz != 224:
        img = img.resize((self.img_siz, self.img_siz)) 
        
        if self.transform:
           img = self.transform(img) # fig2 =plt.figure(); plt.imshow(img.permute(1, 2, 0)); plt.title('transformed image: '+str(img.size()))
        return (img, label, self.file.loc[idx, 'fileName'], self.file.loc[idx, 'img_dir']) 


class SelfCensorship(Dataset):
    def __init__(self, img_siz, file, transform = None):
        self.file      = file
        self.img_siz   = img_siz
        self.transform = transform
            
    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):
        label = self.file.loc[idx, 'label']
        domains = self.file.loc[idx, 'data']
        if 'all_meta_entropy' in self.file.columns:
            uncertainty = self.file.loc[idx, 'all_meta_entropy']
        else:
            uncertainty = -100
            
        if 'all_censor' in self.file.columns:
            do_censor  = self.file.loc[idx, 'all_censor']
        else:
            do_censor = -100
        img_path = os.path.join(self.file.loc[idx, 'img_dir'], self.file.loc[idx, 'fileName'])   
        img = Image.open(img_path).convert('RGB') # .convert('L')
        # if self.img_siz != 224:
        img = img.resize((self.img_siz, self.img_siz)) 
        
        if self.transform:
           img = self.transform(img) # fig2 =plt.figure(); plt.imshow(img.permute(1, 2, 0)); plt.title('transformed image: '+str(img.size()))
        return img, label, img_path, domains, uncertainty, do_censor


class OnlyKeepONH(Dataset):
    def __init__(self, img_siz, file: str, transform=None):
        self.img_siz = img_siz
        self.file = file
        self.transform = transform

    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):
        label = self.file.loc[idx, 'label']
        img_path = os.path.join(self.file.loc[idx, 'img_dir'], self.file.loc[idx, 'fileName'])
        mask_path = os.path.join(self.file.loc[idx, 'img_dir'].replace('image', 'mask_OD'), self.file.loc[idx, 'fileName'])

        imag = Image.open(img_path).convert('RGB')
        # if self.img_siz != 224:
        imag = imag.resize((self.img_siz, self.img_siz))

        # Load the ONH mask as a PIL Image
        onh_mask = Image.open(mask_path).convert('L')
        onh_mask = onh_mask.resize((self.img_siz, self.img_siz))

        # Apply the mask to block the ONH region
        imag = Image.composite(imag, Image.new('RGB', imag.size, (0, 0, 0)), onh_mask)

        if self.transform:
            imag = self.transform(imag)

        if 'patient_Id' in self.file.columns:
            return (imag, label, self.file.loc[idx, 'fileName'], self.file.loc[idx, 'patient_Id'])
        else:
            return (imag, label, self.file.loc[idx, 'fileName'])


class DelONHCustomDataset(Dataset):
    def __init__(self, img_siz, file: str, transform=None):
        self.img_siz = img_siz
        self.file = file
        self.transform = transform

    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):
        label = self.file.loc[idx, 'label']
        img_path = os.path.join(self.file.loc[idx, 'img_dir'], self.file.loc[idx, 'fileName'])
        mask_path = os.path.join(self.file.loc[idx, 'img_dir'].replace('image', 'mask_OD'), self.file.loc[idx, 'fileName'])

        imag = Image.open(img_path).convert('RGB')
        imag = imag.resize((self.img_siz, self.img_siz))

        # Load the ONH mask as a PIL Image
        onh_mask = Image.open(mask_path).convert('L')
        onh_mask = onh_mask.resize((self.img_siz, self.img_siz))

        # Create a copy of the image to work with
        blocked_image = Image.new("RGB", imag.size)

        for x in range(imag.width):
            for y in range(imag.height):
                original_pixel = imag.getpixel((x, y))
                mask_pixel = onh_mask.getpixel((x, y))
                
                # Binarize the mask value
                if mask_pixel > 0:
                    mask_pixel = 1
                else:
                    mask_pixel = 0
        
                # If the binarized mask pixel is 1, set the pixel in the new image to black
                if mask_pixel == 1:
                    blocked_image.putpixel((x, y), (0, 0, 0))
                else:
                    blocked_image.putpixel((x, y), original_pixel)
          
        if self.transform:
            blocked_image = self.transform(blocked_image)
        return (blocked_image, label)
    

class DelONHVessCustomDataset(Dataset):
    def __init__(self, img_siz, file: str, transform=None):
        self.img_siz = img_siz
        self.file = file
        self.transform = transform

    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):
        label = self.file.loc[idx, 'label']
        img_path = os.path.join(self.file.loc[idx, 'img_dir'], self.file.loc[idx, 'fileName'])
        mask_path = os.path.join(self.file.loc[idx, 'img_dir'].replace('image', 'mask_OD_vessel'), self.file.loc[idx, 'fileName'])

        imag = Image.open(img_path).convert('RGB')
        imag = imag.resize((self.img_siz, self.img_siz))

        # Load the ONH mask as a PIL Image
        onh_mask = Image.open(mask_path).convert('L')
        onh_mask = onh_mask.resize((self.img_siz, self.img_siz))

        # Create a copy of the image to work with
        blocked_image = Image.new("RGB", imag.size)

        for x in range(imag.width):
            for y in range(imag.height):
                original_pixel = imag.getpixel((x, y))
                mask_pixel = onh_mask.getpixel((x, y))
                
                # Binarize the mask value
                if mask_pixel > 0:
                    mask_pixel = 1
                else:
                    mask_pixel = 0
        
                # If the binarized mask pixel is 1, set the pixel in the new image to black
                if mask_pixel == 1:
                    blocked_image.putpixel((x, y), (0, 0, 0))
                else:
                    blocked_image.putpixel((x, y), original_pixel)
            
        if self.transform:
            blocked_image = self.transform(blocked_image)
        return (blocked_image, label)
    
    
def fundus_loader(img_sz, path, transf, bs, drop_last, shuffle, num_workers):
    datafile  = pd.read_csv(path) # 'test_annotations.csv'
    dataset   = CustomDataset(img_siz=img_sz, file= datafile, transform= transf)    
    print(len(dataset))
    loader    = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=shuffle, drop_last = drop_last, num_workers=num_workers) 
    return loader


def block_ONH(img_sz, path, transf, bs, drop_last, shuffle, num_workers):
    datafile  = pd.read_csv(path)
    dataset   = DelONHCustomDataset(img_siz=img_sz, file= datafile, transform= transf)    
    print(len(dataset))
    loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=shuffle, drop_last = drop_last, num_workers=num_workers) 
    return loader


def block_ONH_Vess(img_sz, path, transf, bs, drop_last, shuffle, num_workers):
    datafile  = pd.read_csv(path)
    dataset   = DelONHVessCustomDataset(img_siz=img_sz, file= datafile, transform= transf)    
    print(len(dataset))
    loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=shuffle, drop_last = drop_last, num_workers=num_workers) 
    return loader


def selfcensor_loader(img_sz, datafile, transf, bs, drop_last, shuffle, num_workers):
    dataset   = SelfCensorship(img_siz=img_sz, file= datafile, transform= transf)     
    print(len(dataset))
    loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=shuffle, drop_last = drop_last, num_workers=num_workers) 
    return loader




