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

model = nn.Sequential(
    nn.Linear(2, 3, bias = False, device='cuda:0'),
    nn.Linear(3, 5, bias = False, device='cuda:1')  
    )

x = torch.randn(2, 2,  device='cuda:0')

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
