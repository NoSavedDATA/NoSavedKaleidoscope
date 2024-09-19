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
import random

from PIL import Image

import time



start_time = time.time()

print(f"CuDNN: {torch.backends.cudnn.version()}")


##

batch_size=128
download=False

step = 0
max_steps = 64000
lr = 0.1

steps_per_epoch = int(50000/batch_size)
epoch=0

##




def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        #nn.init.kaiming_normal_(m.weight)

    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight, gain=1.41421)
        #nn.init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='B'): # A was used on the original paper, but B is what it is implemented at nsk
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Sequential(nn.Linear(64, num_classes))

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out





class dataset(Dataset):
    def __init__(self, root, transform):
        self.tfms = transform
        self.files = glob.glob(f"{root}/*/*")
        random.shuffle(self.files)

    def __getitem__(self, idx):

        fname = self.files[idx]

        x = Image.open(fname).convert('RGB')
        x = self.tfms(x)

        y = int(fname.split(os.sep)[-2])
        y = torch.tensor(y, dtype=torch.long)

        return x, y

    def __len__(self):
        return len(self.files)



'''

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

train_loader = torch.utils.data.DataLoader(
        dataset(root='/home/nosaveddata/cifar/train/', transform=transforms.Compose([
            transforms.Resize((32,32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop((32,32), 4),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size,
        num_workers=3, pin_memory=True, drop_last=True)

val_loader = torch.utils.data.DataLoader(
        dataset(root='/home/nosaveddata/cifar/test/', transform=transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=100,
        num_workers=0, pin_memory=True)

'''



normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=batch_size, shuffle=True,
        num_workers=3, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=100, shuffle=False,
        num_workers=0, pin_memory=True)




model = ResNet(BasicBlock, [3, 3, 3]).cuda()

optim = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[32000, 48000])

ce = nn.CrossEntropyLoss()

#print(f"{len(val_loader)}")

def evaluate(model, loader):
    
    acc = 0

    for x, y in loader:
        x = x.cuda()
        y = y.cuda()


        x = model(x)

        acc += ((x.argmax(-1))==y).float().mean()


    acc/=len(loader)

    print(f"Acc: {acc}")



# Deactivate normalize for a proper visual inspection.

#to_pil = transforms.ToPILImage() # visual inspect
#class_names = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] # visual inspect
#out_path = 'visu' # visual inspect

while step < max_steps:
    
    for x, y in train_loader:
        x = x.cuda()
        y = y.cuda().long()

        #print(f"{x[4,0]}")
        #print(f"{x.mean(), x.std()}")

        
        #pil_image = to_pil(x[b].cpu()) # visual inspect

        #pil_image.save(f"{out_path}/{step+b}.jpg", format="JPEG", quality=100) # visual inspect
        #print(f"Step {step+b} has class: {class_names[y[b].cpu().item()]}") # visual inspect


        step+=1
        
        x = model(x)

        loss = ce(x, y)
        
        
        loss.backward()

        optim.step()
        optim.zero_grad()

        sched.step()

        if(step%steps_per_epoch)==0:
            epoch+=1
            
            print(f"Epoch {epoch}.")
            #evaluate(model, val_loader)
            
        
        if step>max_steps:
            break

evaluate(model, val_loader)





print(f"Running time: {time.time()-start_time}\n")

