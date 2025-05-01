from __future__ import print_function
import argparse
import csv
import os
import numpy as np
from pytz import timezone
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
from fundusloader import fundus_loader
from normalize_fundus import get_mean_sd_to_normalize_fundus
import random

# import torchvision.models as models

import models
from utils import progress_bar

parser = argparse.ArgumentParser(description='Base model training')
parser.add_argument('--lr', default=1e-1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--model', default="VGG16_BaseModel", type=str, help='model type (default: VGG16_BaseModel)')
parser.add_argument('--name', default='CIFAR10', type=str, help='name of run')
parser.add_argument('--dataset', default='CIFAR10', type=str, help='name of run')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--epoch', default=1, type=int, help='total epochs to run')
parser.add_argument('--no-augment', dest='augment', action='store_false', help='use standard augmentation (default: True)')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
args = parser.parse_args()


def getOutFileName():
    now_CDT = datetime.now(timezone('America/Chicago'))
    current_time = now_CDT.strftime("%y-%m-%d_%H-%M-%S")
    return current_time

exp_name = getOutFileName()

use_cuda = torch.cuda.is_available()

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
best_epoch = 0

if args.seed != 0:
    torch.manual_seed(args.seed)

'''
Processing data
'''
print('==> Preparing data..')

if args.dataset == 'Fundus224':
    img_size = 224
    mean_fun, sd_fun = get_mean_sd_to_normalize_fundus('../dataset/IODA_TVST_ztrain_val_test_annotations.csv', img_size)
    
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=(-180,180), translate=(0.3, 0.3)),
        # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0),
        transforms.RandomCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean_fun, sd_fun)
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean_fun, sd_fun)
        ])

    trainloader = fundus_loader(img_size, '../dataset/IODA_TVST_train_annotations.csv', transform_train, args.batch_size, True, False, 0)
    testloader  = fundus_loader(img_size, '../dataset/IODA_TVST_val_annotations.csv', transform_test, 200, False, False, 0)


'''
Preparing model
'''

print('==> Building model..')
# net = models.resnet50(pretrained=True)
net = models.__dict__[args.model]()

if not os.path.isdir('../results'):
    os.mkdir('../results')
logname = ('../results/Base_' + exp_name + '.csv')

if use_cuda:
    net.cuda()
    print(torch.cuda.device_count())
    cudnn.benchmark = True
    print('Using CUDA..')

criterion = nn.CrossEntropyLoss()
criterion_test = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)

'''
Training model
'''
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    reg_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, vals in enumerate(trainloader):
        if len(vals) > 2:
            inputs, targets, _, _ = vals
        elif len(vals) == 2:
            inputs, targets = vals    
          
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs, _ = net(inputs)
        loss = criterion(outputs, targets)
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar(batch_idx, len(trainloader),'Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)'% (train_loss/(batch_idx+1), reg_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return (train_loss/batch_idx, reg_loss/batch_idx, 100.*correct/total)

'''
Testing model
'''
def validation(epoch):
    global best_acc, best_epoch
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, vals in enumerate(testloader):
            if len(vals) > 2:
                inputs, targets, _, _ = vals
            elif len(vals) == 2:
                inputs, targets = vals    
            
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs, _ = net(inputs)
            loss = criterion_test(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    acc = 100.*correct/total

    if acc > best_acc: # and epoch > 30:
        # checkpoint(acc, epoch)
        best_epoch = epoch
        print(f'acc = {acc} > best acc= {best_acc}')
        best_acc = acc
    return (test_loss/(batch_idx+1), 100.*correct/total)


def checkpoint(acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state(),
    }
    if not os.path.isdir('../checkpoint'):
        os.mkdir('../checkpoint')
    torch.save(state, '../checkpoint/ckpt.t7' + args.name + '_'+ exp_name)


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = args.lr
    if epoch == 100:
        lr /= 10
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['epoch', 'train_loss', 'reg loss', 'train_acc', 'val_loss', 'val_acc', 'best_epoch', 'bs', 'lr', 'wd', 'dataset', 'img_size', 'model', 'train_transform'])

for epoch in range(start_epoch, args.epoch):
    train_loss, reg_loss, train_acc = train(epoch)
    val_loss, val_acc = validation(epoch)
    adjust_learning_rate(optimizer, epoch)
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, train_loss, reg_loss, train_acc.item(), val_loss, val_acc.item(), best_epoch, args.batch_size, args.lr, args.decay, args.dataset, img_size, net.__class__.__name__ , transform_train])




