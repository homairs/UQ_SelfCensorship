#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 17:12:16 2024

@author: homai
"""

import os
import sys
import csv
import argparse
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torchvision.datasets as datasets
import torchvision.transforms as transforms

import preproc as pre
import dataloaders as loaders
from fundusloader import fundus_loader


def save_img(file, noise, severity, saveDir, fname):
    if noise == 'Gaussian':
        trf = gaussian_nois(severity)
    # elif noise == 'impulse_noise':
    #     trf = impulse_noise(severity)
    elif noise == 'permt_nois':
        trf = permt_nois(severity)
    elif noise == 'contrast':
        trf = contrast_noise(severity)
    elif noise == 'speckle':
        trf = speckle_noise(severity)
    elif noise == 'combo_permt_contr':
        sev1 = severity[0]
        sev2 = severity[1]
        trf = combo_permt_contr(sev1, sev2)
    elif noise == 'combo_spec_permt_contr_guas':
        sev1 = severity[0]
        sev2 = severity[1]
        sev3 = severity[2]
        sev4 = severity[3]
        trf = combo_spec_permt_contr_guas(sev1, sev2, sev3, sev4)        
        
    data_loader = fundus_loader(224, file, trf, 140, False, False, 0)
    
    for batch_idx, vals in enumerate(data_loader):
        if len(vals) > 2:
            xs, ys, _, _ = vals
            
        elif len(vals) == 2:
            xs, ys = vals
            
        if batch_idx == 0:
            plt.close(); plt.imshow(xs[0].permute(1, 2, 0)); 
            plt.axis('off')
            plt.tight_layout(); 
            plt.savefig(os.path.join(saveDir, f'{fname}.png'), dpi=300, bbox_inches='tight') #  plt.imshow(xs[0,0,...]);
            break


def gaussian_nois(nois_level):
    return transforms.Compose([transforms.ToTensor(), pre.Fundus_GaussianFilter_test(severity = nois_level)])

# def impulse_noise(nois_level):
#     return transforms.Compose([transforms.ToTensor(), pre.impulse_noise(severity = nois_level)])

def permt_nois(permt_level):
    return transforms.Compose([transforms.ToTensor(), pre.Fundus_PermutationNoise_test(max_permutation_size=permt_level)])

def contrast_noise(nois_level):
    return transforms.Compose([transforms.ToTensor(), pre.Fundus_ContrastRescaling_test(severity = nois_level)])

def speckle_noise(severity):
    return transforms.Compose([transforms.ToTensor(), pre.AddSpeckleNoise_test(mean=0, std=severity)])


def combo_permt_contr(permt_level, contr_level):
    trf = transforms.Compose([
                             transforms.ToTensor(),
                             # pre.impulse_noise(imp_level),
                             pre.Fundus_PermutationNoise_test(max_permutation_size=permt_level),
                             pre.Fundus_ContrastRescaling_test(contr_level),
                             ])
    return trf

def combo_spec_permt_contr_guas(spec_level, permt_level, contr_level, guas_level):
    trf = transforms.Compose([
                             transforms.ToTensor(),
                             pre.AddSpeckleNoise_test(mean=0, std=spec_level),
                             # pre.impulse_noise(imp_level),
                             pre.Fundus_PermutationNoise_test(max_permutation_size=permt_level),
                             pre.Fundus_GaussianFilter_test(guas_level),
                             pre.Fundus_ContrastRescaling_test(contr_level),
                             ])
    return trf





if __name__ == '__main__':
    # path = '/Volumes/homa/Homa/Glaucoma_prediction/3.UQ_project/dataset'
    path = '../dataset'
    save_dir = '../results/IODA_23-11-14_13-54-27__23-11-14_21-55-51'
    data_cvs = os.path.join(path, 'IODA_TVST_test_annotations.csv')
    
    # ============================= save images of Gaussian noise
    noise_levels = [0.8] #[0.1, 0.5, 1.0, 1.5, 2.5]    
    for idx, val in enumerate(noise_levels):
        save_img(data_cvs, 'Gaussian', val, save_dir, f'guassian_{idx}')
        
    # # ============================= save images of impulse noise
    # imp_nois_levels = [.01, .02, .05, .08, .14]
    # for idx, val in enumerate(imp_nois_levels):
    #     save_img(data_cvs, 'impulse_noise', val, save_dir, f'impulse_noise_{idx}')
    # ============================= save images of Permutation noise
    # noise_levels = [0.0002, 0.002, 0.006, 0.01, 0.06, 0.1, 0.2]
    # for idx, val in enumerate(noise_levels):
    #     save_img(data_cvs, 'permt_nois', val, save_dir, f'permt_{idx}')
                  
    # # ============================= save images of Gaussian noise
    # contr_levels = [15, 10, 7, 6, 4, 3, 2]
    # for idx, val in enumerate(contr_levels):
    #     save_img(data_cvs, 'contrast', val, save_dir, f'contrast_{idx}')
        
    # ============================= save images of Gaussian noise
    # speckle_levels = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35]
    # for idx, val in enumerate(speckle_levels):
    #     save_img(data_cvs, 'speckle', val, save_dir, f'speckle_{idx}')
    # # # ============================= save images of Gaussian noise
    # permt_levels = [0.008, 0.02, 0.1, 0.3, 0.5]
    # contr_levels = [0.02, 0.1, 0.3, 0.5,0.9]
    # for idx, val in enumerate(zip(permt_levels, contr_levels)):
    #     save_img(data_cvs, 'combo_permt_contr', val, save_dir, f'combo_permt_contr_{idx}')        
    # ============================= save images of Gaussian noise
    # spec_levels = [0.1, 0.25, 0.40, 0.55, 0.7]
    # permt_levels = [0.008, 0.02,0.1, 0.3, 0.5]
    # contr_levels = [0.02, 0.1, 0.3, 0.5,0.9]
    # gaus_levels = [0.02, 0.2, 0.6, 1.0, 2.5]
    # for idx, val in enumerate(zip(spec_levels, permt_levels, contr_levels, gaus_levels)):
    #     save_img(data_cvs, 'combo_spec_permt_contr_guas', val, save_dir, f'combo_spec_permt_contr_guas_{idx}')        




