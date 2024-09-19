import os, glob

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

from PIL import Image

import time

start_time = time.time()

print(f"CuDNN: {torch.backends.cudnn.version()}")





class dataset(Dataset):
    def __init__(self, path, tfms):

        self.files = glob.glob(path)

        self.tfms = tfms

    def __getitem__(self, idx):

        f = self.files[idx]

        x = Image.open(f)
        x = self.tfms(x)
        
        return x, f

    def __len__(self):
        return len(self.files)



train_tfms = tt.Compose([
                tt.Resize(224),
                tt.ToTensor()
            ])
val_tfms = tt.Compose([
                tt.Resize(224),
                tt.ToTensor()
            ])
    

train_dataset = dataset("/mnt/d/datasets/ImageNet/ILSVRC/Data/CLS-LOC/train/*/*.JPEG", train_tfms)

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


train_dataset = dataset("/mnt/d/datasets/ImageNet/ILSVRC/Data/CLS-LOC/val/*.JPEG", train_tfms)

train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    num_workers=8,
    shuffle=False,
    pin_memory=True
)

for x, y in tqdm.tqdm(train_loader):
    x = x.squeeze(0)
    if x.shape[0] != 3:
        print(f"{x.shape}")
        print(f"{y}")

        os.remove(y[0])