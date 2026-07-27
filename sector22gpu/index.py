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
    nn.Linear(2, 5, bias = False, device='cuda:0') #2*5*4=40 > 5*4=20
    )

x = torch.randn(1, 7,  device='cuda:0')

balance = [3]
devices = ['cuda:0']

#model = GPipe(
    #model, 
    #balance=balance, 
    #devices=devices, 
    #chunks=1, 
    #checkpoint='always'
    #) 

y = checkpoint.checkpoint(
    model,
    input=x,
    determinism_check='none',
    debug=False,
    early_stop=False,
    use_reentrant=False
    )

#y = model(x)

loss = torch.ones_like(y)
y.backward(loss)

torch.cuda.memory._dump_snapshot(f'official_ES_falsesg22{str(dt.datetime.now().timestamp()).replace(".","")}.pickle')

