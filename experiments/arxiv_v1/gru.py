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

import time

start_time = time.time()

print(f"CuDNN: {torch.backends.cudnn.version()}")

PAD_TOK = 0
UNK_TOK = 1

vocab = {}
last_tok = 2

def build_vocab(file, max_size):
    global last_tok

    with open(file, "r") as f:

        for line in f:
            
            if line not in vocab and last_tok<max_size:
                vocab[line.lower().replace(" ", "").replace("\n","")] = last_tok
                last_tok += 1



def left_pad(file, trunc_to):
    
    tokens = []

    with open(file, "r") as f:
        
        for line in f:
            
            if (len(tokens)>=trunc_to):
                break

            words = line.split(" ")

            for word in words:
                if (len(tokens)>=trunc_to):
                    break

                w = word.lower().replace(" ", "").replace("\n","")

                if w in vocab.keys():
                    tokens.append(vocab[w])
                else:
                    tokens.append(1)

    while len(tokens)<trunc_to:
        tokens.insert(0, PAD_TOK)

    return torch.tensor(tokens, dtype=torch.long)


class dataset(Dataset):
    def __init__(self, path, padding=200, vocab_size=32768):

        self.files = glob.glob(path)

        self.padding = padding
        self.vocab_size = vocab_size


    def __getitem__(self, idx):

        f = self.files[idx]

        x = left_pad(f, self.padding)
        
        y = f.split(os.sep)[-2]
        y = int(y)


        return x, y

    def __len__(self):
        return len(self.files)



def init_xavier(module):
    if type(module) in (nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d):
        nn.init.xavier_uniform_(module.weight, gain=1)

        if module.bias is not None:
            nn.init.zeros_(module.bias)

def init_xavier_tanh(module):
    if type(module) in (nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d):
        nn.init.xavier_uniform_(module.weight, gain=1.667)

        if module.bias is not None:
            nn.init.zeros_(module.bias)

def init_gpt(module):
    #print(f"From init_gpt.\nGpt proj linears should have a special weight initialization not implemented here.")
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        #torch.nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        #torch.nn.init.xavier_normal_(module.weight)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)

class GRU(nn.Module):
    def __init__(self, hiddens):
        super().__init__()

        self.Wz = nn.Linear(hiddens, hiddens, bias=False)
        self.Uz = nn.Linear(hiddens, hiddens, bias=False)
        self.Wr = nn.Linear(hiddens, hiddens, bias=False)
        self.Ur = nn.Linear(hiddens, hiddens, bias=False)
        self.Wh = nn.Linear(hiddens, hiddens, bias=False)
        self.Uh = nn.Linear(hiddens, hiddens, bias=False)
        self.W = nn.Linear(hiddens, hiddens, bias=False)
        self.U = nn.Linear(hiddens, hiddens, bias=False)

        self.Wz.apply(init_xavier)
        self.Uz.apply(init_xavier)
        self.Wr.apply(init_xavier_tanh)
        self.Ur.apply(init_xavier_tanh)
        self.Wh.apply(init_xavier)
        self.Uh.apply(init_xavier)
        self.W.apply(init_xavier)
        self.U.apply(init_xavier)


    def forward(self, x, ht):
        
        for i in range(x.shape[1]):
            x_i = x[:,i]
            z = F.sigmoid(self.Wz(x_i)+self.Uz(ht))
            r = F.sigmoid(self.Wr(x_i)+self.Ur(ht))
            h_ = F.tanh(self.Wh(x_i)+self.Uh(ht))
            
            ht = z*ht + (1-z)*h_

        return ht, ht



class RNN(nn.Module):
    def __init__(self, vocab_size, hiddens, out_hiddens, bs):
        super().__init__()

        self.hiddens = hiddens
        self.bs = bs
        self.vocab_size = vocab_size

        #self.embedding = nn.Embedding(vocab_size, hiddens) # with nn.GRU
        self.embedding = nn.Linear(vocab_size, hiddens, bias=False) # from scratch


        #self.gru = nn.GRU(hiddens, hiddens, batch_first=True, bias=False)
        self.gru = GRU(hiddens) # from scratch

        self.classifier = nn.Linear(hiddens, out_hiddens, bias=False)

        self.embedding.apply(init_xavier)
        self.classifier.apply(init_xavier)

    def forward(self, x):
        

        x = F.one_hot(x, self.vocab_size).float() # when using from scratch
        x = self.embedding(x)


        #ht = torch.zeros(1, self.bs, self.hiddens, device="cuda", dtype=torch.float) # with nn.GRU
        ht = torch.zeros(self.bs, self.hiddens, device="cuda", dtype=torch.float) # when using from scratch

        x, ht = self.gru(x, ht)
        
        #x = self.classifier(x[:,-1]) # with nn.GRU 
        x = self.classifier(x) # when using from scratch

        return x




batch_size = 50
val_bs = 100

step = 0
steps_per_epoch = round(25000/batch_size)
max_steps = steps_per_epoch * 4
lr = 1e-3


epoch=0



build_vocab("/mnt/d/datasets/acl_IMDB/vocab.txt", 32768)

print(f"{vocab}")

train_loader = DataLoader(dataset("/mnt/d/datasets/IMDB/train/*/*.txt"),
                          batch_size=batch_size, num_workers=10, pin_memory=True, shuffle=True)
val_loader = DataLoader(dataset("/mnt/d/datasets/IMDB/test/*/*.txt"),
                          batch_size=val_bs, num_workers=10, pin_memory=True, shuffle=False)


steps_per_epoch = len(train_loader)


model = RNN(32768,256,2, batch_size).cuda()
optim = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-4)

ce = nn.CrossEntropyLoss()


def evaluate(model, loader):

    #print(f"EVAL")
    with torch.no_grad():
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

        # Inspect embeddings
        #for i in range(x.shape[0]):
        #    print(f"{x[i]}")

        step+=1
        x = model(x)


        loss = ce(x, y)
        
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

        optim.step()
        optim.zero_grad()


        if step>max_steps:
            break
        



model.bs = val_bs   
evaluate(model, val_loader)

print(f"Running time: {time.time()-start_time}\n")

