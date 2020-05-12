import os
from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np

dirs1 = ["train/", "test/"]
dirs2 = ["covid", 'normal']
dsts = ["train_n/", "test_n/"]



def find_energy_ref():
    energy = []
    for file in os.listdir("ref"):
        img = cv2.imread(os.path.join("ref" ,file)).astype('float')
        img = cv2.resize(img, (280, 280))[28:252, 28:252]
        I, e = find_energy(img)
        energy.append(e)
    energy = np.array(energy)
    e_ref = np.average(energy, axis=0)
    return e_ref

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

def normalization(e_ref, img):

    I_norm = iter_normalization(img, e_ref)
    I_norm = (I_norm - I_norm.min()) * (255/ (I_norm.max() - I_norm.min()))
    I_norm = np.uint8(I_norm)
    return I_norm


e_ref = find_energy_ref()

for dir1, dst in zip(dirs1, dsts):
    for dir2 in dirs2:
        for file in tqdm(os.listdir(os.path.join(dir1, dir2))):
            src = os.path.join(dir1, dir2, file)
            dest = os.path.join(dst, dir2, file)
            img = cv2.imread(src).astype("float")
            img = cv2.resize(img, (280, 280))[28:252, 28:252]
            img_norm = normalization(e_ref, img)
            cv2.imwrite(os.path.join('new', src), img)
            cv2.imwrite(dest, img_norm)
        
    
