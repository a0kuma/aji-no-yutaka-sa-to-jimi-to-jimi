import os
os.environ["CUBLASLT_WORKSPACE_SIZE"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":0:0"

from icecream import ic
ic.configureOutput(includeContext=True)
import sys
import wandb

import torch
import torch.nn as nn
from torchgpipe import GPipe
from torch.utils.checkpoint import checkpoint_sequential

wandb.init(project="https_r13922a13_lab503html_note_20260727", save_code=True)

torch.cuda.memory._record_memory_history()

model = nn.Sequential(
    nn.Linear(7, 3, bias = False, device='cuda:0'),
    nn.Linear(3, 2, bias = False, device='cuda:1'),  
    nn.Linear(2, 3, bias = False, device='cuda:2'),
    #nn.Linear(3, 3, bias = False, device='cuda:3'),#~
    nn.Linear(3, 5, bias = False, device='cuda:1')
    )

x = torch.randn(1, 7,  device='cuda:0')

balance = [1, 1, 1, 1]
devices = ['cuda:0', 'cuda:1', 'cuda:2']#, 'cuda:3', 'cuda:0']

model = GPipe(
    model, 
    balance=balance, 
    devices=devices, 
    chunks=1, 
    checkpoint='never'
    )

y = model(x)

loss = torch.ones_like(y)
y.backward(loss)

torch.cuda.memory._dump_snapshot('sg22.pickle')

wandb.finish()
