import os
os.environ["CUBLASLT_WORKSPACE_SIZE"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":0:0"

from icecream import ic
ic.configureOutput(includeContext=True)

import torch
import torch.nn as nn
from torchgpipe import GPipe
from torch.utils.checkpoint import checkpoint_sequential

torch.cuda.memory._record_memory_history()

model = nn.Sequential(                           #             1*2*4   >
    nn.Linear(2, 3, bias = False, device='cuda'),#2*7*4        > 7*4
    nn.Linear(3, 5, bias = False, device='cuda') #3*5*4        > 5*4 
    )

x = torch.randn(1, 2,  device='cuda')

balance = [1, 1]
devices = ['cuda:0', 'cuda:1']

model = GPipe(
    model, 
    balance=balance, 
    devices=devices, 
    chunks=2, 
    checkpoint='never'
    )

y = model(x)

loss = torch.ones_like(y)
y.backward(loss)

torch.cuda.memory._dump_snapshot('abc.pickle')
