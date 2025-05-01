# Post-hoc Uncertainty Learning using a Dirichlet Meta-Model
This repository contains official implementation of TVST 2025 paper [Robust Uncertainty-Informed Glaucoma Classification Under Data Shift].

## Requirements
- python == 3.8.8
- pytorch == 1.10.0
- torchvision == 0.11.1
- numpy, scipy, sklearn, random, argparse, csv, os, time, sys

## Usage
### Training base-model
- Fundus224
```
python train_base_model.py --model='VGG16_BaseModel_fundus224_pre' --name='Fundus224' --dataset='Fundus224' --lr=1e-2 --seed=0 --decay=1e-4 --epoch=1
```
### Training meta-model
- For different UQ tasks, simply change the "name", such as --name='CIFAR10_OOD' for OOD detection, and --name='CIFAR10_miss' for Misclassfication.
- Fundus224
```
python train_meta_model_combine.py --base_model='VGG16_BaseModel_fundus224_pre' --meta_model='VGG16_MetaModel_fundus224_combine_pre' --name='Fundus224_OOD' --dataset='Fundus224' --lr=1e-2 --seed_trail=0 --decay=1e-4 --epoch=20 --lambda_KL=1e-1 --batch_size 32
```
### Evaluate
- Fundus224
```
python eval_meta_model.py --base_model='VGG16_BaseModel_fundus224_pre' --meta_model='VGG16_MetaModel_fundus224_combine_pre' --name='Fundus224_OOD' --dataset='Fundus224' --im_sz 128
```

### Datasets
- Please manually download publicly available RIMONE-DL, O-RIGA, REFUGE, LAG, Magrabi, Kaggle EyePACS, MESSIDOR-2, IDRiD fundus datasets.
- Please manually download publicly available CIFAR-10, Omniglot, Fashion-MNIST, SVHN, and K-MNIST natural image datasets.


## Reference
This code is based on the following repositories: 
- https://github.com/maohaos2/PosthocUQ.git
- [Mixup](https://github.com/facebookresearch/mixup-cifar10).
- [lula](https://github.com/wiseodd/lula).



