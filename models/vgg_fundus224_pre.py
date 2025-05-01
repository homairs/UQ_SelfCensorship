#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 16:43:22 2023

@author: homai
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

os.environ['TORCH_HOME'] = '/home/homa'

# Load the pre-trained VGG-16 model
pretrained_vgg16 = models.vgg16(pretrained=True)

'''
VGG16 base-model
'''

class VGG16_BaseModel_fundus224_pre(nn.Module):
    def __init__(self):
        super(VGG16_BaseModel_fundus224_pre, self).__init__()

        # Initialize the layers with pre-trained weights from VGG-16
        self.conv1_1 = pretrained_vgg16.features[0]
        self.conv1_2 = pretrained_vgg16.features[2]
        self.batchnorm1 = pretrained_vgg16.features[1]

        self.conv2_1 = pretrained_vgg16.features[5]
        self.conv2_2 = pretrained_vgg16.features[7]
        self.batchnorm2 = pretrained_vgg16.features[6]

        self.conv3_1 = pretrained_vgg16.features[10]
        self.conv3_2 = pretrained_vgg16.features[12]
        self.conv3_3 = pretrained_vgg16.features[14]
        self.batchnorm3 = pretrained_vgg16.features[13]

        self.conv4_1 = pretrained_vgg16.features[17]
        self.conv4_2 = pretrained_vgg16.features[19]
        self.conv4_3 = pretrained_vgg16.features[21]
        self.batchnorm4 = pretrained_vgg16.features[20]

        self.conv5_1 = pretrained_vgg16.features[24]
        self.conv5_2 = pretrained_vgg16.features[26]
        self.conv5_3 = pretrained_vgg16.features[28]
        self.batchnorm5 = pretrained_vgg16.features[27]

        last_pool_output_size = 512 * (224 // (2 ** 5)) ** 2
        self.fc1 = nn.Linear(last_pool_output_size, 256)
        self.fc2 = nn.Linear(256, 2)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.batchnorm1(self.conv1_2(x)))
        x = self.pooling(x)
        fea1 = x.view(x.size(0), -1)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.batchnorm2(self.conv2_2(x)))
        x = self.pooling(x)
        fea2 = x.view(x.size(0), -1)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.batchnorm3(self.conv3_3(x)))
        x = self.pooling(x)
        fea3 = x.view(x.size(0), -1)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.batchnorm4(self.conv4_3(x)))
        x = self.pooling(x)
        fea4 = x.view(x.size(0), -1)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.batchnorm5(self.conv5_3(x)))
        x = self.pooling(x)
        fea5 = x.view(x.shape[0], -1)

        feature = F.relu(self.fc1(fea5))
        predict = self.fc2(feature)
        return predict, [fea1, fea2, fea3, fea4, fea5]


'''
VGG16 meta-model
'''

class VGG16_MetaModel_fundus224_combine_pre(nn.Module):
    def __init__(self, fea_dim1, fea_dim2, fea_dim3, fea_dim4, fea_dim5):
        super(VGG16_MetaModel_fundus224_combine_pre, self).__init__()
        self.pooling = nn.MaxPool1d(kernel_size=2, stride=2)
        # self.classifier1_fc1 = nn.Linear(fea_dim1, 2048)
        # self.classifier1_fc2 = nn.Linear(1024, 512)
        # self.classifier1_fc3 = nn.Linear(256, 10)

        # self.classifier2_fc1 = nn.Linear(fea_dim2, 4096)
        # self.classifier2_fc2 = nn.Linear(2048, 1024)
        # self.classifier2_fc3 = nn.Linear(512, 256)
        # self.classifier2_fc4 = nn.Linear(256, 10)

        self.classifier3_fc1 = nn.Linear(fea_dim3, 2048)
        self.classifier3_fc2 = nn.Linear(1024, 512)
        self.classifier3_fc3 = nn.Linear(512, 256)
        self.classifier3_fc4 = nn.Linear(256, 10)

        self.classifier4_fc1 = nn.Linear(fea_dim4, 1024)
        self.classifier4_fc2 = nn.Linear(512, 256)
        self.classifier4_fc3 = nn.Linear(256, 10)

        self.classifier5_fc1 = nn.Linear(fea_dim5, 256)
        self.classifier5_fc2 = nn.Linear(256, 10)

        self.classifier_final = nn.Linear(3 * 10, 2)

    def forward(self, fea1, fea2, fea3, fea4, fea5):
        
        # print(fea1.size())
        # print(fea2.size())
        # print(fea3.size())
        # print(fea4.size())
        # print(fea5.size())
        
        # fea1 = F.relu(self.classifier1_fc1(fea1))
        # fea1 = self.pooling(fea1)
        # fea1 = F.relu(self.classifier1_fc2(fea1))
        # fea1 = self.pooling(fea1)
        # fea1 = F.relu(self.classifier1_fc3(fea1))

        # fea2 = F.relu(self.classifier2_fc1(fea2))
        # fea2 = self.pooling(fea2)
        # fea2 = F.relu(self.classifier2_fc2(fea2))
        # fea2 = self.pooling(fea2)
        # fea2 = F.relu(self.classifier2_fc3(fea2))
        # fea2 = F.relu(self.classifier2_fc4(fea2))

        fea3 = F.relu(self.classifier3_fc1(fea3))
        fea3 = self.pooling(fea3)
        fea3 = F.relu(self.classifier3_fc2(fea3))
        fea3 = F.relu(self.classifier3_fc3(fea3))
        fea3 = F.relu(self.classifier3_fc4(fea3))

        fea4 = F.relu(self.classifier4_fc1(fea4))
        fea4 = self.pooling(fea4)
        fea4 = F.relu(self.classifier4_fc2(fea4))
        fea4 = F.relu(self.classifier4_fc3(fea4))

        fea5 = F.relu(self.classifier5_fc1(fea5))
        fea5 = F.relu(self.classifier5_fc2(fea5))

        fea = torch.cat((fea3, fea4, fea5), 1) # fea1, fea2, 
        z = self.classifier_final(fea)
        return z
    
    