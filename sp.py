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

pre_process_norm = CustomCompose([transforms.Resize((280, 280)),
                                  transforms.CenterCrop(224),
                                  Normalization(e_ref)])
pre_process = CustomCompose([transforms.Resize((280, 280)),
                             transforms.CenterCrop(224)])
pre_processed = CustomFolder(root='dataset/train', transform=pre_process)
pre_processed_norm = CustomFolder(root='dataset/train', transform=pre_process_norm)

raw = np.array([np.array(image[0], dtype="float") for image in pre_processed)
raw_norm = np.array([np.array(image[0], dtype="float") for image in pre_processed_norm])

unnormalized = CustomCompose([transforms.Resize((280, 280)),
                              transforms.CenterCrop(224),
                              SampleParing(raw, minority_only=False),
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
                                SampleParing(raw_norm, minority_only=False),
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
                                                     
dataset_unnormalized = CustomFolder(root='dataset/train', transform=unnormalized)
dataset_normalized = CustomFolder(root='dataset/train', transform=normalized)
testset_unnormalized = CustomFolder(root='dataset/test', transform=unnormalized_test)
testset_normalized = CustomFolder(root='dataset/test', transform=normalized_test)

DN = DenseNet121().cuda()
VGG = VGG16().cuda()
training(DN, 50,  dataset_unnormalized, testset_unnormalized, "DN-u-smote.pth")
training(VGG, 50, dataset_unnormalized, testset_unnormalized, "vgg-u-smote.pth")

DN = DenseNet121().cuda()
VGG = VGG16().cuda()
training(DN, 50,  data_normalized, testset_normalized, "DN-n-smote.pth")
training(VGG, 50, data_normalized, testset_normalized, "vgg-n-smote.pth")
