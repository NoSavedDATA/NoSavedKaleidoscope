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

from PIL import Image

import time

start_time = time.time()

print(f"CuDNN: {torch.backends.cudnn.version()}")







train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.ToTensor(),
        ]), download=True),
        batch_size=128, shuffle=False,
        num_workers=3, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
        ])),
        batch_size=100, shuffle=False,
        num_workers=0, pin_memory=True)








to_pil = transforms.ToPILImage()

def gen(loader, split="train"):
    counters = {}

    for x, y in loader:
        

        for i in range(x.shape[0]): # across batch axis
            pil_image = to_pil(x[i])
            
            y_ = y[i].cpu().item()

            
            path = f"/home/nosaveddata/cifar/{split}/{y_}"

            os.makedirs(path, mode=0o777, exist_ok=True)

            if y_ not in counters.keys():
                counters[y_] = 0
            else:
                counters[y_] +=1

            path = path+f"/{counters[y_]}.jpg"
            print(f"saving {path}")
            pil_image.save(path, format="JPEG", quality=100)

gen(train_loader)
gen(val_loader, "test")






print(f"Running time: {time.time()-start_time}\n")

