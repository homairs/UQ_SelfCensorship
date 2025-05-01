#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 18:38:38 2024

@author: homai
"""

from __future__ import print_function

import argparse
import os
import csv
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import dataloaders as loaders
from fundusloader import fundus_loader, block_ONH, block_ONH_Vess
from normalize_fundus import get_mean_sd_to_normalize_fundus
import models
import preproc as pre
from metrics import compute_total_entropy, compute_max_prob, compute_differential_entropy, compute_mutual_information, \
    compute_precision, compute_prob
from utils import ROC_OOD, ROC_Selective, convert_to_rgb

parser = argparse.ArgumentParser(description='Meta model Evaluation')
parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--base_model', default="VGG16_BaseModel", type=str, help='model type (default: LeNet)')
parser.add_argument('--base_epoch', default=200, type=int, help='total epochs to train base model')
parser.add_argument('--meta_model', default="VGG16_MetaModel_combine", type=str, help='model type (default: LeNet)')
parser.add_argument('--fea_dim', default=[16384, 8192, 4096, 2048, 512])
parser.add_argument('--name', default='CIFAR10_OOD', type=str, help='name of run')
parser.add_argument('--dataset', default='CIFAR10', type=str, help='name of run')
parser.add_argument('--seed_trail', default=0, type=int, help='random seed')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')
parser.add_argument('--epoch', default=20, type=int, help='total epochs to run')
parser.add_argument('--im_sz', default=224, type=int, help='input image size')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='use standard augmentation (default: True)')
parser.add_argument('--decay', default=5e-4, type=float, help='weight decay')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
use_cuda = torch.cuda.is_available()


base_exp_name = input('What is the name of your BASE model: ')
meta_exp_name = input('What is the name of your META model: ')


if args.dataset == 'Fundus128':
    ratio = int(128/32)*int(128/32) #int(args.img_size/32)*int(args.img_size/32)
    feat_128 = [16384, 8192, 4096, 2048, 512]
    args.fea_dim = [ratio * dim for dim in feat_128]
elif args.dataset == 'Fundus32':
    args.fea_dim = [16384, 8192, 4096, 2048, 512]    
elif args.dataset == 'Fundus224' and args.base_model == 'VGG16_BaseModel_fundus224_pre':
    ratio = int(224/32)*int(224/32) #int(args.img_size/32)*int(args.img_size/32)
    feat_224 = [16384, 8192, 4096, 2048, 512]
    args.fea_dim = [ratio * dim for dim in feat_224]
elif args.dataset == 'Fundus224' and args.base_model == 'ResNet50_BaseModel_fundus224_pre':    
    args.fea_dim = [802816, 401408, 200704, 100352, 18432]   


if use_cuda:
    torch.manual_seed(args.seed_trail)
    torch.cuda.manual_seed_all(args.seed_trail)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed_trail)
    random.seed(args.seed_trail)
    os.environ['PYTHONHASHSEED'] = str(args.seed_trail)

'''
Processing data
'''

def get_fundus_transform(path_to_normalize_imgs):
    mu, sd = get_mean_sd_to_normalize_fundus(path_to_normalize_imgs, args.im_sz) 
    transf = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mu, sd)
                            ])
    return transf

print('==> Preparing data..')
# Noisy validation set for OOD

def gaussian_nois(nois_level):
    return transforms.Compose([transforms.ToTensor(), pre.Fundus_GaussianFilter_test(severity = nois_level)])

def permut_noise(nois_level):
    return transforms.Compose([transforms.ToTensor(), pre.Fundus_PermutationNoise_test(max_permutation_size=nois_level, seed=42)])

def contrast_noise(nois_level):
    return transforms.Compose([transforms.ToTensor(), pre.Fundus_ContrastRescaling_test(severity = nois_level)])

def speckle_noise(severity):
    return transforms.Compose([transforms.ToTensor(), pre.AddSpeckleNoise_test(mean=0, sd=severity, seed=42)])

def combo_permt_gauss_contr(permt_level, guas_level, contr_level):
    trf = transforms.Compose([
                             transforms.ToTensor(),
                             # pre.impulse_noise(imp_level),
                             pre.Fundus_PermutationNoise_test(max_permutation_size=permt_level, seed=42),
                             pre.Fundus_GaussianFilter_test(guas_level),
                             pre.Fundus_ContrastRescaling_test(contr_level),
                             ])
    return trf

def combo_spec_guas(spec_level, guas_level):
    trf = transforms.Compose([
                             transforms.ToTensor(),
                             pre.AddSpeckleNoise_test(mean=0, sd=spec_level, seed=42),
                             pre.Fundus_GaussianFilter_test(guas_level),
                             ])
    return trf

def combo_permt_contr(permt_level, contr_level):
    trf = transforms.Compose([
                             transforms.ToTensor(),
                             pre.Fundus_PermutationNoise_test(max_permutation_size=permt_level, seed=42),
                             pre.Fundus_ContrastRescaling_test(contr_level),
                             ])
    return trf

def combo_permt_gauss(permt_level, guas_level):
    trf = transforms.Compose([
                             transforms.ToTensor(),
                             pre.Fundus_PermutationNoise_test(max_permutation_size=permt_level, seed=42),
                             pre.Fundus_GaussianFilter_test(guas_level),
                             ])
    return trf

def combo_gauss_contr(guas_level, contr_level):
    trf = transforms.Compose([
                             transforms.ToTensor(),
                             pre.Fundus_GaussianFilter_test(guas_level),
                             pre.Fundus_ContrastRescaling_test(contr_level),
                             ])
    return trf



# ================================ # ================================ # ================================ 
# ================================ # ================================ # ================================ 
# ================================ Gaussian noise
# guas1_trf = gaussian_nois(0.4)
# guas2_trf = gaussian_nois(0.6)
# guas3_trf = gaussian_nois(0.8)
# guas4_trf = gaussian_nois(1)
# guas5_trf = gaussian_nois(1.2)
# guas6_trf = gaussian_nois(1.5)
# # ================================ Permutation noise
# permt1_trf  = permut_noise(0.03)  # [0.003, 0.005, 0.007, 0.03, 0.05, 0.7]
# permt2_trf  = permut_noise(0.05)  # [0.01, 0.02,0.03,0.05,0.07,0.09]
# permt3_trf  = permut_noise(0.1)
# permt4_trf  = permut_noise(0.15)
# permt5_trf  = permut_noise(0.3)
# permt6_trf  = permut_noise(0.6)
# # ================================ Contrast noise
# contr1_trf = contrast_noise(3) # [15,10,7,5,3,1], [2,4,6,8,10,12], [8,12,14,16,18,20]
# contr2_trf = contrast_noise(2.5) # WORKED: [3, 2.5, 2, 1, 0.5, 0.1]
# contr3_trf = contrast_noise(2)
# contr4_trf = contrast_noise(1)
# contr5_trf = contrast_noise(0.5)
# contr6_trf = contrast_noise(0.1)
# ================================ Speckle noise
# spec1_trf = speckle_noise(0.1) 
# spec2_trf = speckle_noise(0.15)
# spec3_trf = speckle_noise(0.2)
# spec4_trf = speckle_noise(0.25)
# spec5_trf = speckle_noise(0.3)
# spec6_trf = speckle_noise(0.35)
# ================================ impulse + contrast noise
# permt_gaus_contr1_trf = combo_permt_gauss_contr(0.03,0.4,3)
# permt_gaus_contr2_trf = combo_permt_gauss_contr(0.05,0.6,2.5)
# permt_gaus_contr3_trf = combo_permt_gauss_contr(0.1,0.8,2)
# permt_gaus_contr4_trf = combo_permt_gauss_contr(0.15,1,1)
# permt_gaus_contr5_trf = combo_permt_gauss_contr(0.3,1.2,0.5)
# permt_gaus_contr6_trf = combo_permt_gauss_contr(0.6,1.5,0.1)
# # ================================ speckle+ impulse + guassian + contrast noise
# spec_guas1_trf = combo_spec_guas(0.1, 0.4)
# spec_guas2_trf = combo_spec_guas(0.15, 0.6)
# spec_guas3_trf = combo_spec_guas(0.2, 0.8)
# spec_guas4_trf = combo_spec_guas(0.25, 1)
# spec_guas5_trf = combo_spec_guas(0.3, 1.2)
# spec_guas6_trf = combo_spec_guas(0.35, 1.5)
# # ================================ Permutation + contrast noise
# permt_contr1_trf = combo_permt_contr(0.03, 3)
# permt_contr2_trf = combo_permt_contr(0.05, 2.5)
# permt_contr3_trf = combo_permt_contr(0.1, 2)
# permt_contr4_trf = combo_permt_contr(0.15, 1)
# permt_contr5_trf = combo_permt_contr(0.3, 0.5)
# permt_contr6_trf = combo_permt_contr(0.6, 0.1)
# # ================================ Permutation + Gaussian noise
permt_gaus1_trf = combo_permt_gauss(0.03,0.4)
permt_gaus2_trf = combo_permt_gauss(0.05,0.6)
permt_gaus3_trf = combo_permt_gauss(0.1,0.8)
permt_gaus4_trf = combo_permt_gauss(0.15,1)
permt_gaus5_trf = combo_permt_gauss(0.3,1.2)
permt_gaus6_trf = combo_permt_gauss(0.6,1.5)
# # ================================ Gaussian + contrast noise
gaus_contr1_trf = combo_gauss_contr(0.4,3)
gaus_contr2_trf = combo_gauss_contr(0.6,2.5)
gaus_contr3_trf = combo_gauss_contr(0.8,2)
gaus_contr4_trf = combo_gauss_contr(1,1)
gaus_contr5_trf = combo_gauss_contr(1.2,0.5)
gaus_contr6_trf = combo_gauss_contr(1.5,0.1)



funuds_trans_names = [
                        # 'guas1', 'guas2', 'guas3', 'guas4', 'guas5', 'guas6',
                        # 'permt1', 'permt2', 'permt3', 'permt4', 'permt5', 'permt6',
                        # 'contr1', 'contr2', 'contr3', 'contr4', 'contr5', 'contr6', 
                       # 'spec1', 'spec2','spec3', 'spec4', 'spec5', 'spec6', 
                       # 'permt_gaus_contr1', 'permt_gaus_contr2', 'permt_gaus_contr3', 'permt_gaus_contr4', 'permt_gaus_contr5', 'permt_gaus_contr6',
                       # 'spec_guas1', 'spec_guas2', 'spec_guas3', 'spec_guas4', 'spec_guas5', 'spec_guas6',
                       'permt_gaus1', 'permt_gaus2', 'permt_gaus3', 'permt_gaus4', 'permt_gaus5', 'permt_gaus6',
                       'gaus_contr1', 'gaus_contr2', 'gaus_contr3', 'gaus_contr4', 'gaus_contr5', 'gaus_contr6',
                       # 'permt_contr1', 'permt_contr2', 'permt_contr3', 'permt_contr4', 'permt_contr5', 'permt_contr6',   
                      ]


fundus_transf = [
                   # guas1_trf, guas2_trf, guas3_trf, guas4_trf, guas5_trf, guas6_trf,
                    # permt1_trf, permt2_trf, permt3_trf, permt4_trf, permt5_trf, permt6_trf,
                    # contr1_trf, contr2_trf, contr3_trf, contr4_trf, contr5_trf, contr6_trf,
                  # spec1_trf, spec2_trf, spec3_trf, spec4_trf, spec5_trf, spec6_trf,
                   # permt_gaus_contr1_trf, permt_gaus_contr2_trf, permt_gaus_contr3_trf, permt_gaus_contr4_trf, permt_gaus_contr5_trf, permt_gaus_contr6_trf,
                  # spec_guas1_trf, spec_guas2_trf, spec_guas3_trf, spec_guas4_trf, spec_guas5_trf, spec_guas6_trf,
                  permt_gaus1_trf, permt_gaus2_trf, permt_gaus3_trf, permt_gaus4_trf, permt_gaus5_trf, permt_gaus6_trf,
                  gaus_contr1_trf, gaus_contr2_trf, gaus_contr3_trf, gaus_contr4_trf, gaus_contr5_trf, gaus_contr6_trf,
                  # permt_contr1_trf, permt_contr2_trf, permt_contr3_trf, permt_contr4_trf, permt_contr5_trf, permt_contr6_trf,   
                 ]
# ====================================

IODA_test_trf = get_fundus_transform('../dataset/IODA_TVST_test_annotations.csv')

mu_ref, sd_ref= get_mean_sd_to_normalize_fundus('../dataset/REFUGE_annotations.csv', args.im_sz)
mu_rim, sd_rim= get_mean_sd_to_normalize_fundus('../dataset/RIMEONE-DL_annotations.csv', args.im_sz)
mu_org, sd_org= get_mean_sd_to_normalize_fundus('../dataset/ORIGA_annotations.csv', args.im_sz)
mu_lag, sd_lag= get_mean_sd_to_normalize_fundus('../dataset/LAG_annotations.csv', args.im_sz)
mu_mag, sd_mag= get_mean_sd_to_normalize_fundus('../dataset/Magrabi_annotations.csv', args.im_sz)
mu_gls, sd_gls= get_mean_sd_to_normalize_fundus('../dataset/subset_GLS_annotations.csv', args.im_sz)
mu_kgl, sd_kgl= get_mean_sd_to_normalize_fundus('../dataset/Kaggle_annotations.csv', args.im_sz)
mu_mes, sd_mes= get_mean_sd_to_normalize_fundus('../dataset/Messidor2_annotations.csv', args.im_sz)
mu_idr, sd_idr= get_mean_sd_to_normalize_fundus('../dataset/IDRID_annotations.csv', args.im_sz)


testloader = None
if args.dataset in ['Fundus32', 'Fundus128', 'Fundus224']:
    testloader = fundus_loader(args.im_sz, '../dataset/IODA_TVST_test_annotations.csv', IODA_test_trf, 140, False, False, 0)

'''
Processing OOD data
'''
oodloaders = []
oodnames   = []
if args.dataset in ['Fundus32', 'Fundus128', 'Fundus224']:    
    # ====== Fundus image datasets loaders
    for i, trf in enumerate(fundus_transf):
        oodnames.append(f'REFUGE_{funuds_trans_names[i]}')
        trf = transforms.Compose([*trf.transforms, transforms.Normalize(mu_ref, sd_ref)])
        oodloaders.append(fundus_loader(args.im_sz, '../dataset/REFUGE_annotations.csv', trf, 32, False, False, 0))
    
    for i, trf in enumerate(fundus_transf):
        oodnames.append(f'RIMEONE-DL_{funuds_trans_names[i]}')
        trf = transforms.Compose([*trf.transforms, transforms.Normalize(mu_rim, sd_rim)])
        oodloaders.append(fundus_loader(args.im_sz, '../dataset/RIMEONE-DL_annotations.csv', trf, 32, False, False, 0))
    
    for i, trf in enumerate(fundus_transf):
        oodnames.append(f'ORIGA_{funuds_trans_names[i]}')
        trf = transforms.Compose([*trf.transforms, transforms.Normalize(mu_org, sd_org)])
        oodloaders.append(fundus_loader(args.im_sz, '../dataset/ORIGA_annotations.csv', trf, 32, False, False, 0))

    for i, trf in enumerate(fundus_transf):
        oodnames.append(f'LAG_{funuds_trans_names[i]}')
        trf = transforms.Compose([*trf.transforms, transforms.Normalize(mu_lag, sd_lag)])
        oodloaders.append(fundus_loader(args.im_sz, '../dataset/LAG_annotations.csv', trf, 32, False, False, 0))
    
    for i, trf in enumerate(fundus_transf):
        oodnames.append(f'Magrabi_{funuds_trans_names[i]}')
        trf = transforms.Compose([*trf.transforms, transforms.Normalize(mu_mag, sd_mag)])
        oodloaders.append(fundus_loader(args.im_sz, '../dataset/Magrabi_annotations.csv', trf, 32, False, False, 0))
    
    for i, trf in enumerate(fundus_transf):
        oodnames.append(f'Cropped_GL-S_{funuds_trans_names[i]}')   
        trf = transforms.Compose([*trf.transforms, transforms.Normalize(mu_gls, sd_gls)])
        oodloaders.append(fundus_loader(args.im_sz, '../dataset/IODA_test_annotations_glSuspect.csv', trf, 32, False, False, 0))
        
    for i, trf in enumerate(fundus_transf):
        oodnames.append(f'Kaggle_{funuds_trans_names[i]}')
        trf = transforms.Compose([*trf.transforms, transforms.Normalize(mu_kgl, sd_kgl)])
        oodloaders.append(fundus_loader(args.im_sz, '../dataset/Kaggle_annotations.csv', trf, 32, False, False, 0))
        
    for i, trf in enumerate(fundus_transf):
        oodnames.append(f'MESSIDOR2_{funuds_trans_names[i]}')
        trf = transforms.Compose([*trf.transforms, transforms.Normalize(mu_mes, sd_mes)])
        oodloaders.append(fundus_loader(args.im_sz, '../dataset/Messidor2_annotations.csv', trf, 32, False, False, 0))
    
    for i, trf in enumerate(fundus_transf):
        oodnames.append(f'IDRiD_{funuds_trans_names[i]}')
        trf = transforms.Compose([*trf.transforms, transforms.Normalize(mu_idr, sd_idr)])
        oodloaders.append(fundus_loader(args.im_sz, '../dataset/IDRID_annotations.csv', trf, 32, False, False, 0))



print('current seeds', args.seed_trail)
if use_cuda:
    torch.manual_seed(args.seed_trail)
    torch.cuda.manual_seed_all(args.seed_trail)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed_trail)
    random.seed(args.seed_trail)
    os.environ['PYTHONHASHSEED'] = str(args.seed_trail)
print('==> Resuming from checkpoint..')
assert os.path.isdir('../checkpoint'), 'Error: no checkpoint directory found!'
checkpoint_base = torch.load(
    '../checkpoint/ckpt.t7' + args.dataset + '_' + base_exp_name)
checkpoint_meta = torch.load(
    '../checkpoint/ckpt.t7' + args.name + '_Meta_' + base_exp_name + '_' + meta_exp_name)

if use_cuda:
    if args.dataset in ['Fundus128', 'Fundus32', 'Fundus224']:
        base_net = models.__dict__[args.base_model]().cuda()
    print(torch.cuda.device_count())
    cudnn.benchmark = True
    print('Using CUDA..')
    base_net.load_state_dict(checkpoint_base['net'])
    base_net.eval()
    
    if args.dataset in ['Fundus128', 'Fundus32', 'Fundus224']:
        meta_net = models.__dict__[args.meta_model](fea_dim1=args.fea_dim[0], fea_dim2=args.fea_dim[1],
                                                    fea_dim3=args.fea_dim[2], fea_dim4=args.fea_dim[3],
                                                    fea_dim5=args.fea_dim[4]).cuda()
    meta_net.load_state_dict(checkpoint_meta['meta_net'])
    meta_net.eval()
    print(base_net); print('')
    print(''); print(meta_net)

# Collect all uncertainty scores
def get_uncertainty_score(data_name, loader, label, get_preds=False):
    assert label in [0, 1]
    label = 1. * label

    base_net.eval()
    meta_net.eval()

    labels = []  # is this point ID (0) or OOD (1)?
    base_ents = []
    base_maxps = []
    base_energy = []

    diff_ents = []
    mis = []
    ents = []
    maxps = []
    precs = []

    base_preds = []
    preds = []

    p1 =[] 
    base_p1 = []
    dir_alpha0 = []
    dir_alpha1 = []
    
    ground_truth = []
    fileNames  = []
    imgDirs    = []

    with torch.no_grad():
        # In distribution data
        for batch_idx, vals in enumerate(loader):
            if len(vals) > 2:
                xs, ys, fname, imdir = vals
                fileNames.append(fname)
                imgDirs.append(imdir)
            elif len(vals) == 2:
                xs, ys = vals
                fileNames = []
                imgDirs   = []
                
            if batch_idx == 0:
                plt.close(); plt.imshow(xs[0].permute(1, 2, 0)); plt.tight_layout(); plt.savefig(f'../results/eval_{data_name}.png') #  plt.imshow(xs[0,0,...]);

            if use_cuda:
                xs, ys = xs.cuda(), ys.cuda()

            base_logits, fea_list = base_net(xs)
            logits = meta_net(*fea_list)

            if get_preds:
                # Get predictions (misclassification binary labels)
                _, base_predicted = torch.max(base_logits.data, 1)
                base_wrongs = base_predicted.ne(ys.data)
                _, meta_predicted = torch.max(logits.data, 1)
                meta_wrongs = meta_predicted.ne(ys.data)

                base_preds.append(base_predicted.data.cpu())
                preds.append(meta_predicted.data.cpu())

            # Uncertainty Criterion
            labels.append(label * torch.ones(ys.shape[0]))

            base_ents.append(compute_total_entropy(base_logits).data.cpu())
            base_p1.append(compute_prob(base_logits).data.cpu())
            base_maxps.append(compute_max_prob(base_logits).data.cpu())
            base_energy.append(torch.logsumexp(base_logits, -1).data.cpu())

            diff_ents.append(compute_differential_entropy(logits).data.cpu())
            mis.append(compute_mutual_information(logits).data.cpu())
            ents.append(compute_total_entropy(logits).data.cpu())
            p1.append(compute_prob(logits).data.cpu())
            maxps.append(compute_max_prob(logits).data.cpu())
            precs.append(compute_precision(logits).data.cpu())
            
            ground_truth.append(ys)
            
            dir_alpha0.append(torch.exp(logits)[:, 0].data.cpu())
            dir_alpha1.append(torch.exp(logits)[:, 1].data.cpu())
            
        if get_preds:
            base_preds = torch.cat(base_preds, 0)
            preds = torch.cat(preds, 0)
            
            base_p1 = torch.cat(base_p1, 0)
            p1 = torch.cat(p1, 0)
   
        return torch.cat(diff_ents, 0), \
               torch.cat(mis, 0), \
               torch.cat(ents, 0), \
               torch.cat(maxps, 0), \
               torch.cat(precs,  0), \
               torch.cat(labels,  0), \
               torch.cat(base_ents, 0), torch.cat(base_maxps, 0), torch.cat(base_energy, 0), \
               base_preds, preds, \
               base_p1, p1, \
               torch.cat(dir_alpha0, 0), torch.cat(dir_alpha1, 0), \
               ground_truth, fileNames, imgDirs
               

def prep_data(atensor, maxL):
    tensor_vals  = [tensor.item() for tensor in atensor]  # get values of tensor
    tensor_vals.extend([None] * (maxL - len(tensor_vals)))
    return tensor_vals

def save_to_file(res, colms, out_fname):
    with open(out_fname, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(colms)  # Write the header row
        csv_writer.writerows(res)  



diff_ents, mis, ents, maxps, precs, labels, base_ents, base_maxps, base_energy, base_preds, meta_preds, base_pb1, meta_pb1, meta_alfa0, meta_alfa1, IND_gt, IND_fname, IND_imdir = \
    get_uncertainty_score('IND', testloader, label=0, get_preds=True)

print(f'length of probabilities IND = {(base_pb1).shape}')

# evaluation
if args.name in ['Fundus32_miss', 'Fundus128_miss', 'Fundus224_miss']:
    # Evaluate misclassification performance (for test dataset)
    ROC_Selective(diff_ents, mis, ents, maxps, precs,
                  base_ents, base_maxps,
                  base_preds, meta_preds)

elif args.name in ['Fundus32_OOD', 'Fundus128_OOD', 'Fundus224_OOD']:
    for i in range(len(oodloaders)):
        df = pd.DataFrame()
        print(oodnames[i])
        ood_diff_ents, ood_mis, ood_ents, ood_maxps, ood_precs, ood_labels, \
        ood_base_ents, ood_base_maxps, ood_base_energy, ood_base_preds, ood_meta_preds, ood_base_pb1, ood_meta_pb1, ood_meta_alfa0, ood_meta_alfa1, OOD_gt, OOD_fname, OOD_imdir = \
            get_uncertainty_score(oodnames[i], oodloaders[i], label=1, get_preds=True)
        print(f'length of probabilities OOD = {(ood_base_pb1).shape}')

        all_diff_ents = torch.cat([diff_ents, ood_diff_ents])
        all_mis = torch.cat([mis, ood_mis])
        all_ents = torch.cat([ents, ood_ents])
        all_maxps = torch.cat([maxps, ood_maxps])
        all_precs  = torch.cat([precs, ood_precs])
        all_labels  = torch.cat([labels, ood_labels])
        all_base_ents = torch.cat([base_ents, ood_base_ents])
        all_base_maxps = torch.cat([base_maxps, ood_base_maxps])

        # Evaluate OOD detection performance
        ROC_OOD(all_diff_ents, all_mis, all_ents, all_maxps, all_precs,
                all_labels, all_base_ents, all_base_maxps, '../results/'+oodnames[i]+f'_AUC__{base_exp_name}__{meta_exp_name}.png')

        # =====> saving results in file
        max_len = max(len(maxps), len(ood_maxps))
        print(f'Base len= {len(maxps)}, Dir len= {len(ood_maxps)} ===>>> max len = {max_len}')
        # ================ meta and base stats for IND data
        maxprob       = prep_data(maxps, max_len);       metapreds     = prep_data(meta_preds, max_len)
        entropy       = prep_data(ents, max_len);        baseentropy   = prep_data(base_ents, max_len)
        base_maxprob  = prep_data(base_maxps, max_len);  basepreds     = prep_data(base_preds, max_len)
        base_prob1    = prep_data(base_pb1, max_len);    meta_prob1    = prep_data(meta_pb1, max_len);
        ind_dir_alfa0 = prep_data(meta_alfa0, max_len);  ind_dir_alfa1 = prep_data(meta_alfa1, max_len);
        ind_dir_difent= prep_data(diff_ents, max_len);   ind_dir_mis   = prep_data(mis, max_len);  
        # ================ meta and base stats for OOd data
        ood_maxprob    = prep_data(ood_maxps, max_len);      ood_base_maxprob = prep_data(ood_base_maxps, max_len)
        ood_entropy    = prep_data(ood_ents, max_len);       ood_base_entropy = prep_data(ood_base_ents, max_len)
        ood_metapreds  = prep_data(ood_meta_preds, max_len); ood_basepreds    = prep_data(ood_base_preds, max_len)
        ood_base_prob1 = prep_data(ood_base_pb1, max_len);   ood_meta_prob1   = prep_data(ood_meta_pb1, max_len);
        ood_dir_alfa0  = prep_data(ood_meta_alfa0, max_len); ood_dir_alfa1    = prep_data(ood_meta_alfa1, max_len);
        ood_dir_difent = prep_data(ood_diff_ents, max_len);  ood_dir_mis      = prep_data(ood_mis, max_len);       
        
        # ================
        df = pd.DataFrame({
                           'ind_meta_mi':ind_dir_mis, 'ood_meta_mi':ood_dir_mis, 
                           'ind_meta_diffent':ind_dir_difent, 
                           'ood_meta_diffent':ood_dir_difent, 'ind_meta_entropy':entropy, 'ood_meta_entropy':ood_entropy, 
                           'ind_base_entropy':baseentropy, 'ood_base_entropy':ood_base_entropy, 'ind_meta_preds':metapreds, 
                           'ind_base_preds':basepreds, 'ind_meta_maxp':maxprob, 'ind_base_maxp':base_maxprob, 'ind_meta_prob1':meta_prob1, 
                           'ind_base_prob1':base_prob1, 'ood_meta_preds':ood_metapreds, 'ood_meta_maxp':ood_maxprob, 'ood_meta_prob1':ood_meta_prob1, 
                           'ood_base_preds':ood_basepreds, 'ood_base_maxp':ood_base_maxprob, 'ood_base_prob1': ood_base_prob1, 'ind_alfa0': ind_dir_alfa0, 
                           'ind_alfa1': ind_dir_alfa1, 'ood_alfa0': ood_dir_alfa0, 'ood_alfa1': ood_dir_alfa1})
        
        ind_labl = [item.item() for sublist in IND_gt for item in sublist]
        ind_labl.extend([None] * (max_len - len(ind_labl)))
        df['ind_labels'] = ind_labl

        ood_labl = [item.item() for sublist in OOD_gt for item in sublist]
        ood_labl.extend([None] * (max_len - len(ood_labl)))
        df['ood_labels'] = ood_labl

        ind_fnames = [item for sublist in IND_fname for item in sublist]
        ind_fnames.extend([None] * (max_len - len(ind_fnames)))
        df['ind_fileName'] = ind_fnames
        
        ind_imgdir = [item for sublist in IND_imdir for item in sublist]
        ind_imgdir.extend([None] * (max_len - len(ind_imgdir)))
        df['ind_imgDirs'] = ind_imgdir

        ood_fnames = [item for sublist in OOD_fname for item in sublist]
        ood_fnames.extend([None] * (max_len - len(ood_fnames)))
        df['ood_fileName'] = ood_fnames
        
        ood_imgdir = [item for sublist in OOD_imdir for item in sublist]
        ood_imgdir.extend([None] * (max_len - len(ood_imgdir)))
        df['ood_imgDirs'] = ood_imgdir
        

        df.to_csv(f'../results/{oodnames[i]}_probs.csv', index=False)
       







