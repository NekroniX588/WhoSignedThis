import matplotlib.pyplot as plt
import os
import faiss
from tqdm import tqdm
import pandas as pd
import numpy as np
import skimage
import random
import itertools
import logging
import time

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('recognizer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def preprocessing(img, use_tresholding=True, use_delete_noise=True):
    logger.debug("Starting image preprocessing")
    start_time = time.time()
    
    # tresholding
    if use_tresholding:
        logger.debug("Applying thresholding")
        img_threshold = filters.threshold_yen(img)
        img[img>img_threshold] = 255
        img[img<img_threshold] = 0
        
    # delete noise
    if use_delete_noise:
        logger.debug("Removing noise")
        selem =  morphology.disk(1)
        res = morphology.black_tophat(img, selem)
        img = img + res
    
    img = Image.fromarray(img).convert('RGB')
    
    processing_time = time.time() - start_time
    logger.debug(f"Preprocessing completed in {processing_time:.2f} seconds")
    return img

image_transform_test = transforms.Compose([
    transforms.Resize((155, 220)),
    ImageOps.invert,
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

def gen_embedding(net, data, is_gpu=False):
    logger.debug("Generating embedding")
    start_time = time.time()
    
    if is_gpu:
        data = data.cuda()
    
    embedded = net(data)
    embedded_numpy = embedded.data.cpu().numpy()
    
    processing_time = time.time() - start_time
    logger.debug(f"Embedding generation completed in {processing_time:.2f} seconds")
    return embedded_numpy

def calculate_signature(path, net):
    logger.info(f"Calculating signature for image: {path}")
    start_time = time.time()
    
    try:
        logger.debug("Loading and preprocessing image")
        query_img = image_transform_test(preprocessing(skimage.io.imread(path, as_gray=True))).unsqueeze(0)
        
        logger.debug("Generating embedding")
        embedding = gen_embedding(net, query_img, False).reshape(-1)
        
        processing_time = time.time() - start_time
        logger.info(f"Signature calculation completed in {processing_time:.2f} seconds")
        return embedding
    except Exception as e:
        logger.error(f"Error calculating signature: {str(e)}", exc_info=True)
        raise

class TripletNet(nn.Module):
    """Triplet Network."""
    
    def __init__(self):
        """Triplet Network Builder."""
        super(TripletNet, self).__init__()
        logger.info("Initializing TripletNet model")
        
        self.embeddingnet = models.mobilenet_v2(False)
        self.embeddingnet.classifier = nn.Linear(1280, 512)
        
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=8, padding=1,
                                     stride=16)  # 1st sub sampling
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=4, padding=1)

        self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=8, padding=4,
                                     stride=32)  # 2nd sub sampling
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=7, stride=2, padding=3)

        self.dense_layer = torch.nn.Linear(in_features=(512 + 1152*2), out_features=512)
        logger.info("TripletNet model initialized successfully")

    def forward(self, a):
        """Forward pass."""
        logger.debug("Starting forward pass")
        start_time = time.time()
        
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
        
        merge_subsample = torch.cat([first_input, second_input], 1)
        merge_conv = torch.cat([merge_subsample, conv_input], 1)
        final_input = self.dense_layer(merge_conv)
        final_norm = final_input.norm(p=2, dim=1, keepdim=True)
        embedded_a = final_input.div(final_norm.expand_as(final_input))
        
        processing_time = time.time() - start_time
        logger.debug(f"Forward pass completed in {processing_time:.2f} seconds")
        return embedded_a

logger.info("Loading TripletNet model...")
Identificator = TripletNet()
try:
    Identificator.load_state_dict(torch.load('./model/Epoch_30.pt', map_location=torch.device('cpu'))['model'])
    Identificator.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}", exc_info=True)
    raise