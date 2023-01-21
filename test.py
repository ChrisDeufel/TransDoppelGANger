import torch
import os
import logging
import sys
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
x = np.arange(0, 10, 1)
y = x[:len(x)]

print('hello')