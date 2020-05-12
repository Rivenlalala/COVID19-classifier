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

minority = np.array([np.array(image[0], dtype="float") for image in pre_processed if image[1]==0])
minority_norm = np.array([np.array(image[0], dtype="float") for image in pre_processed_norm if image[1]==0])

data = CustomCompose([Majority(minority),
                      transforms.RandomHorizontalFlip(),
                      transforms.RandomRotation(10),
                      transforms.ToTensor(),
                      transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                           std=[0.5, 0.5, 0.5])])

data_norm = CustomCompose([Majority(minority_norm),
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
VGG = VGG16().cuda()
training(DN, 50,  dataset_unnormalized, testset_unnormalized, "DN-u-M.pth")
training(VGG, 50, dataset_unnormalized, testset_unnormalized, "vgg-u-M.pth")

DN = DenseNet121().cuda()
VGG = VGG16().cuda()
training(DN, 50,  dataset_normalized, testset_normalized, "DN-n-M.pth")
training(VGG, 50, dataset_normalized, testset_normalized, "vgg-n-M.pth")
