from __future__ import print_function

import argparse
import os
import sys
import csv
import random
import time
import numpy as np
from pytz import timezone
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn import metrics

import models
import preproc as pre
from losses import BeliefMatchingLoss
from metrics import compute_total_entropy, compute_max_prob, compute_differential_entropy, compute_mutual_information, \
    compute_precision
from utils import progress_bar, convert_to_rgb
from fundusloader import fundus_loader
from normalize_fundus import get_mean_sd_to_normalize_fundus

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='Meta model training')
parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--base_model', default="WideResNet_BaseModel", type=str, help='model type (default: LeNet)')
parser.add_argument('--base_epoch', default=200, type=int, help='total epochs to train base model')
parser.add_argument('--meta_model', default="WideResNet_MetaModel_combine", type=str,
                    help='model type (default: LeNet)')
parser.add_argument('--name', default='CIFAR100_OOD', type=str, help='name of run')
parser.add_argument('--dataset', default='CIFAR100', type=str, help='name of run')
parser.add_argument('--seed_trail', default=0, type=int, help='random seed')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--epoch', default=1, type=int, help='total epochs to run')
parser.add_argument('--saving_epoch', default=1000, type=int, help='total epochs to run')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='use standard augmentation (default: True)')
parser.add_argument('--decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--lambda_KL', default=1e-3, type=float, help='lambda for KL term in ELBO loss')

parser.add_argument('--permt_noise', action='store_true', help='Add permutation noise to the validation (default: FALSE)')
parser.add_argument('--gauss_noise',  action='store_true', help='Add Gaussian noise to the validation (default: FALSE)')
parser.add_argument('--spckl_noise',  action='store_true', help='Add Speckle noise to the validation (default: FALSE)')
parser.add_argument('--contr_noise', action='store_true', help='Add Contrast rescaling noise to the validation (default: FALSE)')
parser.add_argument('--SP_noise', action='store_true', help='Add salt & pepper noise to the validation (default: FALSE)')

parser.add_argument('--noise_level', default='minimal', type=str, choices=['minimal', 'low', 'moderate', 'high'], help='noise level')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
use_cuda = torch.cuda.is_available()

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
best_auroc = 0

def getOutFileName():
    now_CDT = datetime.now(timezone('America/Chicago'))
    current_time = now_CDT.strftime("%y-%m-%d_%H-%M-%S")
    return current_time

base_exp_name = input('What is the name of your BASE model: ')
meta_exp_name = getOutFileName()

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
print('==> Resuming from checkpoint..')
assert os.path.isdir('../checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('../checkpoint/ckpt.t7' + args.dataset + '_' + base_exp_name,
                        map_location=torch.device('cpu') if not use_cuda else None)

'''
Processing data
'''
print('==> Preparing data..')
# Noisy validation set for OOD
if args.dataset == 'Fundus224':
    img_size = 224
    mean_fun, sd_fun = get_mean_sd_to_normalize_fundus('../dataset/IODA_TVST_ztrain_val_test_annotations.csv', img_size) 
    
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        # transforms.RandomAffine(degrees=(20, 20), translate=(0.0, 0.0)),
        # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0),
        transforms.ToTensor(),
        # transforms.Normalize(mean_fun, sd_fun)
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean_fun, sd_fun)
        ])
    
    # transform_noise = transforms.Compose([
    #                         transforms.ToTensor(),
    #                         transforms.Normalize(mean_fun, sd_fun),
    #                         # pre.AddSpeckleNoise(mean=0, std=0.05),
    #                         pre.Fundus_PermutationNoise(),
    #                         pre.Fundus_GaussianFilter(), # sigma = 0.3 + 0.5 * torch.rand(1).item()
    #                         pre.Fundus_ContrastRescaling(), # gamma = 5 + 15 * torch.rand(1).item()
    #                   ])


    meta_net = models.__dict__[args.meta_model](fea_dim1=args.fea_dim[0], fea_dim2=args.fea_dim[1],
                                                fea_dim3=args.fea_dim[2], fea_dim4=args.fea_dim[3],
                                                fea_dim5=args.fea_dim[4])
    

    
    transform_noise = transforms.Compose([transforms.ToTensor()])
    if args.permt_noise:
        transform_noise = transforms.Compose([*transform_noise.transforms, pre.Fundus_PermutationNoise(args.noise_level)])
    if args.gauss_noise:
        transform_noise= transforms.Compose([*transform_noise.transforms, pre.Fundus_GaussianFilter(args.noise_level)])
    if args.spckl_noise:
        transform_noise= transforms.Compose([*transform_noise.transforms, pre.Fundus_AddSpeckleNoise(args.noise_level)])
    if args.contr_noise:
        transform_noise= transforms.Compose([*transform_noise.transforms, pre.Fundus_ContrastRescaling(args.noise_level)])
    if args.SP_noise:
        transform_noise= transforms.Compose([*transform_noise.transforms, pre.Fundus_impulse_noise(args.noise_level)])
    
    
    # transform_noise= transforms.Compose([*transform_noise.transforms, transforms.Normalize(mean_fun, sd_fun),])
    print(transform_noise)

    trainloader = fundus_loader(224, '../dataset/IODA_TVST_train_annotations.csv', transform_train, args.batch_size, True, True, 0)
    valloader = fundus_loader(224, '../dataset/IODA_TVST_val_annotations.csv', transform_test, 140, False, False, 0)
    valloader_noise = fundus_loader(224, '../dataset/IODA_TVST_val_annotations.csv', transform_noise, 140, False, False, 0)
    testloader = fundus_loader(224, '../dataset/IODA_TVST_test_annotations.csv', transform_test, 140, False, False, 0)   


'''
Preparing model
'''
if args.dataset in ['Fundus128', 'Fundus224']:
    base_net = models.__dict__[args.base_model]()

# if args.dataset in ['Fundus128', 'Fundus224']:
#     meta_net = models.__dict__[args.meta_model](fea_dim1=args.fea_dim[0], fea_dim2=args.fea_dim[1],
#                                                 fea_dim3=args.fea_dim[2], fea_dim4=args.fea_dim[3],
#                                                 fea_dim5=args.fea_dim[4])
    
if use_cuda:
    print('Using CUDA..')
    print(torch.cuda.device_count())
    cudnn.benchmark = True
    base_net = base_net.cuda()
    meta_net = meta_net.cuda()

base_net.load_state_dict(checkpoint['net'])
base_net.eval()
for k, v in base_net.named_parameters():
    v.requires_grad = False

param_group = []
meta_net.eval()

optimizer = optim.SGD(meta_net.parameters(), momentum=0.9, weight_decay=args.decay, lr=args.lr)

if not os.path.isdir('../results/'):
    os.mkdir('../results/')
logname = ('../results/Meta_' + base_exp_name + '__'+ meta_exp_name + '.csv')

'''
Training Meta-model
'''
vi_loss = BeliefMatchingLoss(args.lambda_KL, 1)


def compute_logits_and_loss(xs, ys, compute_loss=False):
    loss = torch.Tensor([0])
    _, fea_list = base_net(xs)
    logits = meta_net(*fea_list)

    if compute_loss:
        loss = vi_loss(logits, ys)

    return logits, loss


def train(epoch):
    print('\nEpoch: %d' % epoch)
    base_net.eval()
    meta_net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, vals in enumerate(trainloader):
        if len(vals) > 2:
            xs, ys, _, _ = vals
        elif len(vals) == 2:
            xs, ys = vals    
        
        total += ys.size(0)
        if use_cuda:
            xs, ys = xs.cuda(), ys.cuda()

        logits, loss = compute_logits_and_loss(xs, ys, compute_loss=True)

        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
       
        _, predicted = torch.max(logits.data, 1)
        correct += predicted.eq(ys).cpu().sum()

        progress_bar(current=batch_idx,
                     total=len(trainloader),
                     msg='Loss: %.3f |  Acc: %.3f%% (%d/%d)' % (
                         train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    train_loss_final = train_loss / batch_idx
    acc = 100. * correct / total

    return (train_loss_final, acc)


'''
Testing Meta-model
'''


def test(epoch):
    global best_acc
    base_net.eval()
    meta_net.eval()
    test_loss = 0
    correct = 0

    total_entropy = 0
    max_prob = 0
    mutual_info = 0
    diff_entropy = 0
    precision = 0

    total = 0
    with torch.no_grad():
        for batch_idx, vals in enumerate(testloader):
            if len(vals) > 2:
                xs, ys, _, _ = vals
            elif len(vals) == 2:
                xs, ys = vals
            
            total += ys.size(0)
            if use_cuda:
                xs, ys = xs.cuda(), ys.cuda()

            logits, loss = compute_logits_and_loss(xs, ys, compute_loss=True)
            test_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            correct += predicted.eq(ys.data).cpu().sum()

            # Uncertainty Criterion
            total_entropy += compute_total_entropy(logits).sum()
            max_prob += compute_max_prob(logits).sum()
            mutual_info += compute_mutual_information(logits).sum()
            diff_entropy += compute_differential_entropy(logits).sum()
            precision += compute_precision(logits).sum()

            progress_bar(batch_idx, len(testloader),
                         'Loss: %.3f | Acc: %.3f%% (%d/%d) | '
                         'DEnt: %.3f | MI: %.3f | TotEnt: %.3f | MaxP: %.3f | Prec: %.3f' %
                         (test_loss / (batch_idx + 1), 100. * correct / total, correct, total,
                          diff_entropy / total, mutual_info / total,
                          total_entropy / total, max_prob / total, precision / total))

        test_loss_final = test_loss / total
        acc = 100. * correct / total

        return (test_loss_final, acc)


'''
Validation for OOD task
'''
def UQ_validation():
    global best_auroc
    base_net.eval()
    meta_net.eval()
    flag = True
    total = 0
    val_loss=0
    with torch.no_grad():
        # In distribution data
        for batch_idx, vals in enumerate(valloader):
            if len(vals) > 2:
                xs, ys, _, _ = vals
            elif len(vals) == 2:
                xs, ys = vals
            
            if use_cuda:
                xs, ys = xs.cuda(), ys.cuda()
            
            total += ys.size(0)
            if epoch == 0 and batch_idx == 0:
                plt.close(); plt.imshow(xs[20].cpu().permute(1, 2, 0)); plt.tight_layout(); plt.savefig(f'../results/meta_val_clean.png') #  plt.imshow(xs[0,0,...]);    

            logits, los = compute_logits_and_loss(xs, ys, compute_loss=True)
            val_loss += los.item()

            # Uncertainty Criterion
            mutual_info = compute_mutual_information(logits)
            _, meta_predicted = torch.max(logits.data, 1)
            meta_correct = meta_predicted.ne(ys.data)
            if flag:
                all_label = torch.zeros((ys.size()[0]))
                all_mutual_info = mutual_info.data.cpu()
                all_meta_predicted = meta_correct.data.cpu()
                flag = False
            else:
                all_label = torch.cat((all_label, torch.zeros((ys.size()[0]))), 0)
                all_mutual_info = torch.cat((all_mutual_info, mutual_info.data.cpu()), 0)
                all_meta_predicted = torch.cat((all_meta_predicted, meta_correct.data.cpu()), 0)

        # Out of distribution data
        for batch_idx, vals in enumerate(valloader_noise):
            if len(vals) > 2:
                xs, ys, _, _ = vals
            elif len(vals) == 2:
                xs, ys = vals
                
            if batch_idx == 0:
                plt.close(); plt.imshow(xs[20].cpu().permute(1, 2, 0)); plt.tight_layout(); plt.savefig(f'../results/meta_val_noisy_{args.noise_level}.png') #  plt.imshow(xs[0,0,...]);
                
            
            if use_cuda:
                xs, ys = xs.cuda(), ys.cuda()
            logits, _ = compute_logits_and_loss(xs, ys, compute_loss=False)

            # Uncertainty Criterion
            mutual_info = compute_mutual_information(logits)
            all_label = torch.cat((all_label, torch.ones((ys.size()[0]))), 0)
            all_mutual_info = torch.cat((all_mutual_info, mutual_info.data.cpu()), 0)

    # ood Auroc score evaluated using mutual information
    auroc_MI = metrics.roc_auc_score(all_label.numpy(), all_mutual_info.numpy())
    # if auroc_MI > best_auroc and epoch > 5: # and epoch == (args.epoch - 1):
    # if epoch == 69:
    #     print(f'auroc_MI= {auroc_MI} > best_auroc= {best_auroc}')
    #     checkpoint(auroc_MI, epoch)
    #     best_auroc = auroc_MI

    val_loss_final = val_loss / total
    return auroc_MI, val_loss_final


'''
Inference the meta-model on OOD dataset (noisy images)
'''


def OOD(epoch):
    base_net.eval()
    meta_net.eval()

    total_entropy = 0
    max_prob = 0
    mutual_info = 0
    diff_entropy = 0
    precision = 0

    total = 0
    with torch.no_grad():
        for batch_idx, vals in enumerate(valloader_noise):
            if len(vals) > 2:
                xs, ys, _, _ = vals
            elif len(vals) == 2:
                xs, ys = vals
            
            total += ys.size(0)
            if use_cuda:
                xs, ys = xs.cuda(), ys.cuda()

            # test meta model
            logits, _ = compute_logits_and_loss(xs, ys, compute_loss=False)

            # Uncertainty Criterion
            total_entropy += compute_total_entropy(logits).sum()
            max_prob += compute_max_prob(logits).sum()
            mutual_info += compute_mutual_information(logits).sum()
            diff_entropy += compute_differential_entropy(logits).sum()
            precision += compute_precision(logits).sum()
            progress_bar(batch_idx, len(valloader_noise),
                         'DEnt: %.3f | MI: %.3f | TotEnt: %.3f | MaxP: %.3f | Prec: %.3f'
                         % (diff_entropy / total, mutual_info / total,
                            total_entropy / total, max_prob / total, precision / total))
    return


def checkpoint(auroc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'meta_net': meta_net.state_dict(),
        'auroc': auroc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('../checkpoint'):
        os.mkdir('../checkpoint')
    torch.save(state, '../checkpoint/ckpt.t7' + args.name + '_Meta_' + base_exp_name + '_' + meta_exp_name)


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = args.lr/ 10#np.sqrt(epoch)
    print(f' =========>>>>>>>>>>>>>>> Decreased lr to = {lr}')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_lambda_KL(epoch):
    """increase the KL lambda"""
    # denom = (args.lambda_KL)** (-1) + (epoch - 1) * 5
    # lamda = epoch / denom
    lamda = args.lambda_KL / 10

    v_i_loss = BeliefMatchingLoss(lamda, 1)
    return lamda, v_i_loss



'''
Main training process
'''
if __name__ == '__main__':
    time_start = time.perf_counter()
    
    if not os.path.exists(logname):
        with open(logname, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(['epoch', 'train_loss', 'test_loss', 'val_loss', 'train_acc', 'test_acc', 'AUC_MI_vals', 'lr', 'wd', 'lambda', 'batch_size', 'dataset', 'train_transform', 'val_noise_transform'])

    
    lambda_KL = args.lambda_KL
    lr        = args.lr
    for epoch in range(start_epoch, args.epoch + 1):
        train_loss, train_acc = train(epoch)
        test_loss, test_acc = test(epoch)
        if args.name in ['Fundus128_OOD', 'Fundus224_OOD']:
            OOD(epoch)
            auc_MI_vals, val_loss = UQ_validation()
        
        # if epoch == 10:
        #     lambda_KL, vi_loss = adjust_lambda_KL(epoch)
            # lr = adjust_learning_rate(optimizer, epoch)

        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch, train_loss, test_loss, val_loss, train_acc.item(), test_acc.item(), auc_MI_vals, lr, args.decay, lambda_KL, args.batch_size, args.dataset, transform_train, transform_noise])
    

    print('Finished')
    training_time = time.perf_counter() - time_start
    print('Total training time', training_time)


        
        
        
