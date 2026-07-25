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
    nn.Linear(2, 7, bias = False, device='cuda'),
    nn.Linear(7, 3, bias = False, device='cuda'),
    nn.Linear(3, 5, bias = False, device='cuda')
    )

x = torch.randn(1, 2,  device='cuda')


loss = torch.ones_like(y)
y.backward(loss)

torch.cuda.memory._dump_snapshot('abc.pickle')
