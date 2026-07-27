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

model = nn.Sequential(
    nn.Linear(7, 3, bias = False, device='cuda:0'),#7*3*4=84 > 3*4=12
    nn.Linear(3, 2, bias = False, device='cuda:0'),#3*2*4=24 > 2*4=8
    nn.Linear(2, 7, bias = False, device='cuda:1'),#2*7*4=56 > 7*4=28
    nn.Linear(7, 5, bias = False, device='cuda:1'),#7*5*4=140 > 5*4=20
    )

x = torch.randn(1, 7,  device='cuda:0')

balance = [2, 2]
devices = ['cuda:0','cuda:1']

model = GPipe(
    model, 
    balance=balance, 
    devices=devices, 
    chunks=1, 
    checkpoint='always'
    ) 

y = model(x)

loss = torch.ones_like(y)
y.backward(loss)

torch.cuda.memory._dump_snapshot(f'mama{str(dt.datetime.now().timestamp()).replace(".","")}.pickle')

