import pandas as pd
from shutil import copyfile
import os
pneumonias = ["SARS", "MERS", "ARDS", "Streptococcus", "Pneumocystis", "Klebsiella", "Chlamydophila", "Legionella"]
csv = pd.read_csv("dataset/covid-chestxray-dataset/metadata.csv")
idx_pa = csv["view"].isin(["PA"])
csv = csv[idx_pa]
idx_pneumonias = csv["finding"].isin(pneumonias)
pneu = csv[idx_pneumonias]

l = pneu['filename'].tolist()
for file in l:
  src = os.path.join('dataset/covid-chestxray-dataset/images', file)
  dest = os.path.join('dataset/pneu', file)
  copyfile(src, dest)
