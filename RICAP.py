import os
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch
import pandas as pd
import numpy as np
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
from utils import *
from models import *
import cv2
from tqdm import tqdm, trange


pre_processed = CustomFolder(root='dataset/train',transform=CustomCompose([]))
pre_processed_norm = CustomFolder(root='dataset/train_n', transform=CustomCompose([]))

raw = np.array([np.array(image[0], dtype="float") for image in pre_processed])
raw_norm = np.array([np.array(image[0], dtype="float") for image in pre_processed_norm])

data = CustomCompose([RICAP(pre_processed),
                      transforms.RandomHorizontalFlip(),
                      transforms.RandomRotation(10),
                      transforms.ToTensor(),
                      transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                           std=[0.5, 0.5, 0.5])])

data_norm = CustomCompose([RICAP(pre_processed_norm),
                           transforms.RandomHorizontalFlip(),
                           transforms.RandomRotation(10),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                std=[0.5, 0.5, 0.5])])
test = CustomCompose([transforms.ToTensor(),
                      transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                           std=[0.5, 0.5, 0.5])])
                                                     
dataset_unnormalized = CustomFolder(root='dataset/train', transform=data)
dataset_normalized = CustomFolder(root='dataset/train_n', transform=data_norm)
testset_unnormalized = CustomFolder(root='dataset/test', transform=test)
testset_normalized = CustomFolder(root='dataset/test_n', transform=test)

DN = DenseNet121().cuda()
training(DN, 50,  dataset_unnormalized, testset_unnormalized, "DN-u-R.pth")
VGG = VGG16().cuda()
training(VGG, 50, dataset_unnormalized, testset_unnormalized, "vgg-u-R.pth")

DN = DenseNet121().cuda()
training(DN, 50,  dataset_normalized, testset_normalized, "DN-n-R.pth")
VGG = VGG16().cuda()
training(VGG, 50, dataset_normalized, testset_normalized, "vgg-n-R.pth")