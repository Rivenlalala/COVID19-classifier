import numpy as np
from PIL import Image
from random import random


class Majority():

    def __init__(self, raw_dataset):
        self.dataset = raw_dataset
    
    def __call__(self, sample):
        if sample[1] == 0:
            return sample
        else:
            img = np.array(sample[0], dtype="float")
            rand_img = self.dataset[np.random.randint(len(self.dataset))]
            new = img + (rand_img - img) * random()
            new = Image.fromarray(np.uint8(new))

            return [new, 0]


class RICAP():

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
        for i in idx:
            rand_img.append(np.array(self.dataset[i][0]))
            label += self.dataset[i][1]
        label /= 4

        w, h = np.random.beta(self.beta, self.beta, 2)
        w1 = int(size * w)
        h1 = int(size * h)
        w_ = [w1, size - w1, w1, size - w1]
        h_ = [h1, h1, size - h1, size - h1]
        locx = [np.random.randint(size - w1), np.random.randint(w1),
                np.random.randint(size - w1), np.random.randint(w1)]
        locy = [np.random.randint(size - h1), np.random.randint(size - h1),
                np.random.randint(h1), np.random.randint(h1)]
        
        cat = [np.array(sample[0])[locx[0]:locx[0] + w_[0], locy[0]:locy[0] + h_[0]]]
        for i, img in enumerate(rand_img):
            cat.append(img[locx[i + 1]:locx[i + 1] + w_[i + 1], locy[i + 1]:locy[i + 1] + h_[i + 1]])
        img = np.hstack((np.vstack((cat[0], cat[1])), np.vstack((cat[2], cat[3]))))
        img = Image.fromarray(img)

        return [img, label]
        

class SampleParing():
    def __init__(self, raw_dataset, minority_only):
        self.dataset = raw_dataset
        self.minority_only = minority_only

    def __call__(self, sample):
        if ((self.minority_only & sample[1] == 1) | (random() > 0.5)) :
            return sample
        else:
            img = np.array(sample[0], dtype="float")
            rand_img = self.dataset[np.random.randint(len(self.dataset))]
            
            new = (img + rand_img) / 2
            new = Image.fromarray(np.uint8(new))
            return [new, sample[1]]
            

class Smote():

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
        if (sample[1] == 1 | random() > 0.5):
            return sample
        else:
            img = np.array(sample[0], dtype="float64")
            new = img
            nearests = self.knn(img)
            for nearest in nearests:
                new += random() * (nearest - img)
            #new = (new - new.min()) / (new.max() - new.min()) * 255
            new = Image.fromarray(np.uint8(new))
            return [new, 0]
