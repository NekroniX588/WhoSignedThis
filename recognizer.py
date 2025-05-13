import matplotlib.pyplot as plt
import os
import faiss
from tqdm import tqdm
import pandas as pd
import numpy as np
import skimage
import random
import itertools

from PIL import Image
from PIL import ImageOps

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import models
from torchvision import transforms

import skimage
from skimage import filters
from skimage import morphology

from skimage.transform import resize
from skimage.io import imread
from skimage import img_as_ubyte
from pathlib import Path

def preprocessing(img, use_tresholding=True, use_delete_noise=True):
    # tresholding
    if use_tresholding:
        img_threshold = filters.threshold_yen(img)
        img[img>img_threshold] = 255
        img[img<img_threshold] = 0
        
    # delete noise
    if use_delete_noise:
        selem =  morphology.disk(1)
        res = morphology.black_tophat(img, selem)
        img = img + res
    
    img = Image.fromarray(img).convert('RGB')
    
    return img

image_transform_test = transforms.Compose([
    transforms.Resize((155, 220)),
    ImageOps.invert,
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

def gen_embedding(net, data, is_gpu=False):
    if is_gpu:
        data = data.cuda()
    # data = Variable(data)

    embedded = net(data)
    embedded_numpy = embedded.data.cpu().numpy()
    return embedded_numpy


def calculate_signature(path, net):
    query_img = image_transform_test(preprocessing(skimage.io.imread(path, as_gray=True))).unsqueeze(0)
    embedding = gen_embedding(net, query_img, False).reshape(-1)
    return embedding
  
class TripletNet(nn.Module):
    """Triplet Network."""

    def __init__(self):
        """Triplet Network Builder."""
        super(TripletNet, self).__init__()
#         self.embeddingnet = models.mobilenet_v2()
#         self.embeddingnet.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=True)
#         self.embeddingnet.classifier = nn.Linear(1280, 4096)
        self.embeddingnet = models.mobilenet_v2(True)
        self.embeddingnet.classifier = nn.Linear(1280, 512)
        
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=8, padding=1,
                                     stride=16)  # 1st sub sampling
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=4, padding=1)

        self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=8, padding=4,
                                     stride=32)  # 2nd sub sampling
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=7, stride=2, padding=3)

        self.dense_layer = torch.nn.Linear(in_features=(512 + 1152*2), out_features=512)

    def forward(self, a):
        """Forward pass."""
        # anchor
        conv_input = self.embeddingnet(a)
        conv_norm = conv_input.norm(p=2, dim=1, keepdim=True)
        conv_input = conv_input.div(conv_norm.expand_as(conv_input))

        first_input = self.conv1(a)
        first_input = self.maxpool1(first_input)
        first_input = first_input.view(first_input.size(0), -1)
        first_norm = first_input.norm(p=2, dim=1, keepdim=True)
        first_input = first_input.div(first_norm.expand_as(first_input))
        
        second_input = self.conv2(a)
        second_input = self.maxpool2(second_input)
        second_input = second_input.view(second_input.size(0), -1)
        second_norm = second_input.norm(p=2, dim=1, keepdim=True)
        second_input = second_input.div(second_norm.expand_as(second_input))
        
        merge_subsample = torch.cat([first_input, second_input], 1)  # batch x (3072)

        merge_conv = torch.cat([merge_subsample, conv_input], 1)  # batch x (4096 + 3072)

        final_input = self.dense_layer(merge_conv)
        final_norm = final_input.norm(p=2, dim=1, keepdim=True)
        embedded_a = final_input.div(final_norm.expand_as(final_input))

        return embedded_a
      
      
Identificator = TripletNet()
Identificator.load_state_dict(torch.load('./model/Epoch_30.pt', map_location=torch.device('cpu'))['model'])
Identificator.eval()
print('DONE')