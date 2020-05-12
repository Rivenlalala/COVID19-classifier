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


e_ref = find_energy_ref()

unnormalized = CustomCompose([transforms.Resize((280, 280)),
                            transforms.CenterCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomRotation(10),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                 std=[0.5, 0.5, 0.5])])
unnormalized_test = CustomCompose([transforms.Resize((280, 280)),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                     std=[0.5, 0.5, 0.5])])
normalized = CustomCompose([transforms.Resize((280, 280)),
                                transforms.CenterCrop(224),
                                Normalization(e_ref),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(10),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                     std=[0.5, 0.5, 0.5])])
normalized_test = CustomCompose([transforms.Resize((280, 280)),
                                transforms.CenterCrop(224),
                                Normalization(e_ref),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                     std=[0.5, 0.5, 0.5])])
dataset_unnormalized = CustomFolder(root='dataset/train_s', transform=unormalized)
dataset_normalized = CustomFolder(root='dataset/train_s', transform=normalized)
testset_unnormailized = CustomFolder(root='dataset/test_s', transform=unnormalized_test)
testset_normalized = CustomFolder(root='dataset/test_s', transform=normalized_test)

DN = DenseNet121().cuda()
VGG = VGG16().cuda()
training(DN, 50,  dataset_unnormalized, testset_unnormalized, "DN-u-c.pth")
training(VGG, 50, dataset_unnormalized, testset_unnormalized, "vgg-u-c.pth")

DN = DenseNet121().cuda()
VGG = VGG16().cuda()
training(DN, 50,  data_normalized, testset_normalized, "DN-n-c.pth")
training(VGG, 50, data_normalized, testset_normalized, "vgg-n-c.pth")
