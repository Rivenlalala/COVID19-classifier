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


data = CustomCompose([transforms.RandomHorizontalFlip(),
                      transforms.RandomRotation(10),
                      transforms.ToTensor(),
                      transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                 std=[0.5, 0.5, 0.5])])
test = CustomCompose([transforms.ToTensor(),
                      transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                           std=[0.5, 0.5, 0.5])])


dataset_unnormalized = CustomFolder(root='dataset/train', transform=data)
dataset_normalized = CustomFolder(root='dataset/train_n', transform=data)
testset_unnormalized = CustomFolder(root='dataset/test', transform=test)
testset_normalized = CustomFolder(root='dataset/test_n', transform=test)

DN = DenseNet121().cuda()
VGG = VGG16().cuda()
training(DN, 50,  dataset_unnormalized, testset_unnormalized, "DN-u-n.pth")
training(VGG, 50, dataset_unnormalized, testset_unnormalized, "vgg-u-n.pth")

DN = DenseNet121().cuda()
VGG = VGG16().cuda()
training(DN, 50,  dataset_normalized, testset_normalized, "DN-n-n.pth")
training(VGG, 50, dataset_normalized, testset_normalized, "vgg-n-n.pth")
