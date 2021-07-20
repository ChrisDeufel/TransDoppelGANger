import torch
import os
import logging
import sys
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

x = torch.reshape(torch.arange(150), (5, 10, 3))
y = x.view(-1, x.size(1) * x.size(2))
print('hello')