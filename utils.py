from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from PIL import Image
from random import random
import os
import cv2
import torch
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm, trange
import copy


class CustomTransforms():
    pass


class CustomDataset(ConcatDataset):

    def __init__(self, datasets, transform=None):
        super().__init__(datasets)
        self.transform = transform

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        if self.transform is not None:
            sample = self.transform((sample))

        return sample


class Majority(CustomTransforms):

    def __init__(self, raw_dataset):
        self.dataset = raw_dataset
    
    def __call__(self, sample):
        if sample[1] == 0:
            return sample
        else:
            img = np.array(sample[0], dtype="float")
            rand_img = self.dataset[np.random.randint(len(self.dataset))]
            new = img + (rand_img - img) / 2
            new = Image.fromarray(np.uint8(new))

            return new, 0


class RICAP(CustomTransforms):

    def __init__(self, dataset, beta=1):
        self.dataset = dataset
        self.beta = beta

    def __call__(self, sample):
        '''
        (w1, h1)  (w2, h1)
        (w1, h2)  (w2, h2)
        '''
        size = 224
        idx = np.random.randint(len(self.dataset), size=3)
        rand_img = []
        label = sample[1]


        w, h = np.random.beta(self.beta, self.beta, 2)
        w1 = int(size * w)
        h1 = int(size * h)
        w2 = size - w1
        h2 = size - h1
        w_ = [w1, w2, w1, w2]
        h_ = [h1, h1, h2, h2]
        locx = [np.random.randint(w2), np.random.randint(w1),
                np.random.randint(w2), np.random.randint(w1)]
        locy = [np.random.randint(h2), np.random.randint(h2),
                np.random.randint(h1), np.random.randint(h1)]
        areas = [w2*h1, w1*h2, w2*h2]
        label *= w1 * h1 / size ** 2
        for i, area in zip(idx, areas):
            rand_img.append(np.array(self.dataset[i][0]))
            label += self.dataset[i][1] * area / size ** 2
        
        cat = [np.array(sample[0])[locy[0]:locy[0] + h_[0], locx[0]:locx[0] + w_[0]]]
        for i, img in enumerate(rand_img):
            cat.append(img[locy[i + 1]:locy[i + 1] + h_[i + 1], locx[i + 1]:locx[i + 1] + w_[i + 1]])
        img = np.vstack((np.hstack((cat[0], cat[1])), np.hstack((cat[2], cat[3]))))
        img = Image.fromarray(img)

        return img, label
        

class SampleParing(CustomTransforms):
    def __init__(self, raw_dataset, minority_only):
        self.dataset = raw_dataset
        self.minority_only = minority_only

    def __call__(self, sample):
        img = np.array(sample[0], dtype="float")
        rand_img = self.dataset[np.random.randint(len(self.dataset))]

        new = (img + rand_img) / 2
        new = Image.fromarray(np.uint8(new))
        return new, sample[1]
            

class Smote(CustomTransforms):

    def __init__(self, raw_dataset, k):
        self.dataset = raw_dataset
        self.k = k

    def knn(self, img):
        img = img.reshape(-1)
        data = self.dataset.reshape(len(self.dataset), -1)
        distance = np.linalg.norm(img - data, axis=1)
        nearest = np.argsort(distance)

        return self.dataset[nearest[:self.k]]

    def __call__(self, sample):
        if sample[1] == 1:
            return sample
        else:
            img = np.array(sample[0], dtype="float64")
            new = img
            nearests = self.knn(img)
            for nearest in nearests:
                new += random() * (nearest - img)
            new = Image.fromarray(np.uint8(new))
            return new, 0


class CustomFolder(ImageFolder):
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform((sample, target))

        return sample
    

class CustomCompose(transforms.Compose):
    
    def __call__(self, sample):
        img = sample[0]
        label = sample[1]
        for t in self.transforms:
            if isinstance(t, CustomTransforms):
                img, label = t((img, label))
            else:
                img = t(img)
        return img, label
        

def find_energy(img):
    L = []
    G = [img]
    for i in range(5):
        G.append(cv2.GaussianBlur(img, (0, 0), sigmaX=2**i))
        L.append(G[i] - G[i + 1])
    L.append(G[-1])
    L = np.array(L)
    e = np.std(L, axis=(1, 2))
    e = e[:, 0]
    return L, e

def iter_normalization(img, e_ref):
    I_norm = img
    for i in range(10):
        I, e = find_energy(I_norm)
        #I_norm = np.sum((e_ref / e).reshape(-1, 1, 1, 1) * I, axis=0)
        I_norm = np.zeros(I_norm.shape)
        for level, layer in enumerate(I):
            I_norm += e_ref[level] / e[level] * I[level]
    return I_norm

def find_energy_ref():
    energy = []
    for file in os.listdir("dataset/ref"):
        img = cv2.imread(os.path.join("dataset/ref" ,file)).astype('float')
        img = cv2.resize(img, (280, 280))[28:252, 28:252]
        I, e = find_energy(img)
        energy.append(e)
    energy = np.array(energy)
    e_ref = np.average(energy, axis=0)
    return e_ref


class Normalization(CustomTransforms):

    def __init__(self, e_ref):
        self.e_ref = e_ref
    
    def __call__(self, sample):
        img = np.array(sample[0]).astype(float)
        I_norm = iter_normalization(img, self.e_ref)

        I_norm = (I_norm - I_norm.min()) * (255/ (I_norm.max() - I_norm.min()))
        I_norm = Image.fromarray(np.uint8(I_norm))
        return I_norm, sample[1]

def print_acc(model, dataloader, testloader):

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    model = model.eval()
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            predicted = torch.zeros_like(outputs, dtype=torch.long)
            predicted[outputs >= 0.5] = 1
            for i, pred in enumerate(predicted):
                if (pred == labels[i]) & (pred == 1):
                    TN += 1
                elif (pred == labels[i]) & (pred == 0):
                    TP += 1
                elif (pred != labels[i]) & (pred == 0):
                    FP += 1
                else:
                    FN += 1

    print("Training Acc: ",)
    print('TP: %d  TN: %d  FP: %d  FN: %d' % (TP, TN, FP, FN))

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            predicted = torch.zeros_like(outputs, dtype=torch.long)
            predicted[outputs >= 0.5] = 1
            for i, pred in enumerate(predicted):
                if (pred == labels[i]) & (pred == 1):
                    TN += 1
                elif (pred == labels[i]) & (pred == 0):
                    TP += 1
                elif (pred != labels[i]) & (pred == 0):
                    FP += 1
                else:
                    FN += 1
    print("Testing Acc: ",)
    print('TP: %d  TN: %d  FP: %d  FN: %d' % (TP, TN, FP, FN))

def validation(model, dataloader):
    
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    model = model.eval()
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            predicted = torch.zeros_like(outputs, dtype=torch.long)
            predicted[outputs >= 0.5] = 1
            for i, pred in enumerate(predicted):
                if (pred == labels[i]) & (pred == 1):
                    TN += 1
                elif (pred == labels[i]) & (pred == 0):
                    TP += 1
                elif (pred != labels[i]) & (pred == 0):
                    FP += 1
                else:
                    FN += 1
    acc = (TN + TP) / (TN + TP + FN + FP)
    model.train()
    return acc

def training(model, epoch, dataset, validate, testset, filename):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
    valloader = DataLoader(validate, batch_size=8, shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=8, shuffle=False, num_workers=4)
    best_acc = 0
    with trange(epoch) as t:
        for epoch in t:
            t.set_description('EPOCH %i' % (epoch + 1))
            running_loss = 0
            for i, data in enumerate(dataloader, 0):
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.type(torch.FloatTensor).cuda()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(1), labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            running_loss /= i
            t.set_postfix(loss=running_loss)
            if epoch > 90:
                acc = validation(model, valloader)
                if acc > best_acc:
                    best_model = copy.deepcopy(model)
                    best_acc = acc
    print('Finished Training')
    model = best_model
    torch.save(model.state_dict(), os.path.join('models', filename))
    print_acc(model, dataloader, testloader)
