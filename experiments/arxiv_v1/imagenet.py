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

import tqdm
import random

import time

start_time = time.time()

print(f"CuDNN: {torch.backends.cudnn.version()}")





def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        #nn.init.kaiming_normal_(m.weight)

    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight, gain=1.41421)
        #nn.init.kaiming_normal_(m.weight)




def init_xavier(module):
    if type(module) in (nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d):
        nn.init.xavier_uniform_(module.weight, gain=1)

        if module.bias is not None:
            nn.init.zeros_(module.bias)

def init_relu(module):
    if type(module) in (nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d):
        nn.init.xavier_uniform_(module.weight, gain=1.41421)

        if module.bias is not None:
            nn.init.zeros_(module.bias)

class Residual_Block(nn.Module):
    def __init__(self, in_channels, channels, stride=1, act=nn.ReLU(), out_act=nn.ReLU(), norm=True, init=init_relu, bias=False):
        super().__init__()
        
        
        conv1 = nn.Sequential(nn.Conv2d(in_channels, channels, kernel_size=3, padding=1,
                                            stride=stride, bias=bias),
                              nn.BatchNorm2d(channels, eps=1e-6),
                              act)
        conv2 = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=bias),
                              nn.BatchNorm2d(channels, eps=1e-6),
                              out_act)

        conv1.apply(init)
        conv2.apply(init if out_act!=nn.Identity() else init_xavier)
        
        self.conv = nn.Sequential(conv1, conv2)
        
        self.proj=nn.Identity()
        if stride>1 or in_channels!=channels:
            self.proj = nn.Sequential(nn.Conv2d(in_channels, channels, kernel_size=1, padding=0,
                                                stride=stride, bias=bias),
                                      nn.BatchNorm2d(channels)
                                      )

        
        self.proj.apply(init_relu)
        self.out_act = out_act
        
    def forward(self, X):
        
        Y = self.conv(X)
        return Y+self.proj(X)
    




class ResNet18(nn.Module):
    def __init__(self, in_channels, num_classes=1000):
        super().__init__()
        

        self.conv1 = nn.Sequential(
                                nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False), 
                                nn.MaxPool2d(3, 2, 1),
                                nn.BatchNorm2d(64),
                                nn.ReLU())


        self.conv2 = nn.Sequential(Residual_Block(64,  64,  1), Residual_Block(64,  64))
        self.conv3 = nn.Sequential(Residual_Block(64,  128, 2), Residual_Block(128, 128))
        self.conv4 = nn.Sequential(Residual_Block(128, 256, 2), Residual_Block(256, 256))
        self.conv5 = nn.Sequential(Residual_Block(256, 512, 2), Residual_Block(512, 512))
        
        self.net = nn.Sequential(self.conv2, self.conv3, self.conv4, self.conv5)
        

        self.classifier = nn.Sequential(nn.AvgPool2d(7),
                                        nn.Flatten(), 
                                        nn.Linear(512, num_classes, bias=False))


        self.conv1.apply(init_relu)
        self.classifier.apply(init_xavier)

        
    def forward(self, x):
        x = self.conv1(x)

        x = self.net(x)


        x = self.classifier(x)
        

        return x









class dataset(Dataset):
    def __init__(self, path, tfms):

        self.files = glob.glob(path)
        self.tfms = tfms
        random.shuffle(self.files)

    def __getitem__(self, idx):

        f = self.files[idx]

        x = Image.open(f)
        x = self.tfms(x)
        
        y = f.split(os.sep)[-2]
        y = int(y)

        

        return x, y

    def __len__(self):
        return len(self.files)



normalize = tt.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]
                        )



train_tfms = tt.Compose([
                tt.Resize(224),
                tt.RandomHorizontalFlip(),
                tt.RandomCrop(224, 4),
                tt.ToTensor(),
                normalize
            ])
val_tfms = tt.Compose([
                tt.Resize(224),
                tt.RandomCrop(224, 0), # Ensure it is not bugged
                tt.ToTensor(),
                normalize
            ])









model = ResNet18(3).cuda()


step = 0
max_steps = 10000
lr = 0.1

batch_size = 256

steps_per_epoch = int(1281167/batch_size)
epoch=0

optim = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_steps)

ce = nn.CrossEntropyLoss()


train_loader = DataLoader(dataset("/mnt/d/datasets/ImageNet/train/*/*.JPEG", train_tfms),
                          batch_size=batch_size, num_workers=12, pin_memory=True)
                          
val_loader = DataLoader(dataset("/mnt/d/datasets/ImageNet/val/*/*.JPEG",  val_tfms),
                          batch_size=100, num_workers=3, pin_memory=True)

print(len(train_loader))
print(len(val_loader))


def evaluate(model, loader):

    #model.eval() # nsk does not deactivate batchnorm at eval

    acc = 0

    for x, y in tqdm.tqdm(loader):
        x = x.cuda()
        y = y.cuda()

        x = model(x)

        acc += ((x.argmax(-1))==y).float().mean()

    acc/=len(loader)

    print(f"Acc: {acc}")




while step < max_steps:
    for x, y in tqdm.tqdm(train_loader):
        x = x.cuda()
        y = y.cuda()

        #print(f"{y}")

        x = model(x)

        loss = ce(x, y)
        

        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        optim.step()
        optim.zero_grad()

        sched.step()


        '''
        if(step%steps_per_epoch)==0:
            epoch+=1
            
            print(f"Epoch {epoch}.")
        '''

        step+=1
        if step>max_steps:
            break
        



evaluate(model, val_loader)

print(f"Running time: {time.time()-start_time}\n")

