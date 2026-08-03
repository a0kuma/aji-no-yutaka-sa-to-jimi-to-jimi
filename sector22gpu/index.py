import os
os.environ["CUBLASLT_WORKSPACE_SIZE"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":0:0"

from icecream import ic
ic.configureOutput(includeContext=True)
import sys
import datetime as dt
import wandb

import torch
import torch.nn as nn
from torchgpipe import GPipe
from torch.utils import checkpoint
from torch.utils.checkpoint import checkpoint_sequential

torch.cuda.memory._record_memory_history()

x = torch.randn(1, 7,  device='cuda:0')            #1*7*4=28
model = nn.Sequential(
    nn.Linear(7, 2, bias = False, device='cuda:0'),#7*2*4=56 > 2*4=8
    nn.Linear(2, 2, bias = False, device='cuda:0'),
    nn.Linear(2, 3, bias = False, device='cuda:0'),#2*3*4=24 > 3*4=12
    nn.Linear(3, 5, bias = False, device='cuda:0'),#3*5*4=60 > 5*4=20
    nn.Linear(5, 5, bias = False, device='cuda:0'),
    nn.Linear(5, 2, bias = False, device='cuda:0'),#5*2*4=40 > 2*4=8
    nn.Linear(2, 7, bias = False, device='cuda:1'),#2*7*4=56 > 7*4=28
    nn.Linear(7, 7, bias = False, device='cuda:1'),
    nn.Linear(7, 3, bias = False, device='cuda:1'),#7*3*4=84 > 3*4=12 
    nn.Linear(3, 2, bias = False, device='cuda:1'),#3*2*4=24 > 2*4=8
    nn.Linear(2, 2, bias = False, device='cuda:1'),
    nn.Linear(2, 5, bias = False, device='cuda:1'),#2*5*4=40 > 5*4=20
    )

balance = [2, 2, 2, 2, 2, 2]
devices = ['cuda:0', 'cuda:0', 'cuda:0', 'cuda:1', 'cuda:1', 'cuda:1']

model = GPipe(
    model, 
    balance=balance, 
    devices=devices, 
    chunks=1, 
    checkpoint='always'
    ) 

for step in range(2):
    model.zero_grad(set_to_none=False)
    y = model(x)
    loss = torch.ones_like(y)
    y.backward(loss)

torch.cuda.memory._dump_snapshot(f'loop2set_to_none=False_{str(dt.datetime.now().timestamp()).replace(".","")}.pickle')

