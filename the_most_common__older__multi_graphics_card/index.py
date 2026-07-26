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
    nn.Linear(2, 7, bias = False, device='cuda'),#2*7*4        > 7*4
    nn.Linear(7, 3, bias = False, device='cuda'),#7*3*4        > 3*4  
    nn.Linear(3, 5, bias = False, device='cuda'),#3*5*4        > 5*4

    nn.Linear(5, 2, bias = False, device='cuda'),#5*2*4        > 2*4
    nn.Linear(2, 3, bias = False, device='cuda'),#2*3*4        > 3*4
    nn.Linear(3, 5, bias = False, device='cuda'),#3*5*4        > 5*4

    nn.Linear(5, 7, bias = False, device='cuda'),#5*7*4        > 7*4
    nn.Linear(7, 3, bias = False, device='cuda'),#7*3*4        > 3*4
    nn.Linear(3, 5, bias = False, device='cuda') #3*5*4        > 5*4 
    )

x = torch.randn(1, 2,  device='cuda')

y = checkpoint_sequential(
    model, 
    segments=3, 
    input=x,
    use_reentrant=False
    )

loss = torch.ones_like(y)
y.backward(loss)

torch.cuda.memory._dump_snapshot('abc.pickle')
