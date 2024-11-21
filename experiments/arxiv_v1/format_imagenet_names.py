import os, glob, shutil
import sys

import torch
import torchvision
import tarfile
import torch.nn as nn
import torch.nn.functional as F

from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
import torchvision.transforms as tt
from torchvision import transforms
from torch.utils.data import random_split
from torchvision.utils import make_grid

import matplotlib.pyplot as plt
import numpy as np

import tqdm

import pandas as pd

from PIL import Image
import pickle

import time

start_time = time.time()

print(f"CuDNN: {torch.backends.cudnn.version()}")






image_folders = glob.glob("/mnt/d/datasets/ImageNet/ILSVRC/Data/CLS-LOC/train/*")


folder_to_id = {}
label_to_id = {}

i=0
for folder in image_folders:
    folder_to_id[folder] = i
    label_to_id[folder.split(os.sep)[-1]] = i
    i+=1

print(f"{label_to_id}")


i=0
for img in tqdm.tqdm(glob.glob("/mnt/d/datasets/ImageNet/ILSVRC/Data/CLS-LOC/train/*/*.JPEG")):
    label = img.split(os.sep)[-2]



    img_name = img.split(os.sep)[-1]

    label_path = f"/mnt/d/datasets/ImageNet/train/{label_to_id[label]}"

    print(f"making dir {label_path}")
    os.makedirs(label_path, mode=0o777, exist_ok=True)

    print(f"New path: {label_path}/{img_name}")

    try:
        shutil.move(img, f"{label_path}/{img_name}")
        #pass
    except:
        pass

    i+=1



df = pd.read_csv("/mnt/d/datasets/ImageNet/LOC_val_solution.csv")

print(f"df length: {len(df)}")


i=0
for _, row in tqdm.tqdm(df.iterrows()):
    
    img_name = f"{row[0]}.JPEG"

    path = f"/mnt/d/datasets/ImageNet/ILSVRC/Data/CLS-LOC/val/{img_name}"
    label = row[1].split(' ')[0]




    print(f"\n\n{path}\n{label}")

    label_path = f"/mnt/d/datasets/ImageNet/val/{label_to_id[label]}"
    os.makedirs(label_path, mode=0o777, exist_ok=True)

    print(f"New path: {label_path}/{img_name}")
    try:
        shutil.move(path, f"{label_path}/{img_name}")
        #print(f"moving {path} into {label_path}/{img_name}")
    except:
        pass

    i+=1















'''
class dataset(Dataset):
    def __init__(self, path):

        self.files = glob.glob(path)
        
    def __getitem__(self, idx):
        
        return  self.files[idx]

    def __len__(self):
        return len(self.files)


train_dataset = dataset("/mnt/d/datasets/ImageNet/ILSVRC/Data/CLS-LOC/train/*/*.JPEG")

train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    num_workers=8,
    shuffle=False,
    pin_memory=True
)

for f in tqdm.tqdm(train_loader):

    pass



train_dataset = dataset("/mnt/d/datasets/ImageNet/ILSVRC/Data/CLS-LOC/val/*.JPEG")

train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    num_workers=8,
    shuffle=False,
    pin_memory=True
)

for x, y in tqdm.tqdm(train_loader):
    x = x.squeeze(0)  # Remove the batch dimension added by DataLoader
    if x.shape[0] != 3:
        print(f"{x.shape}")
        print(f"{y}")

        os.remove(y[0])
'''