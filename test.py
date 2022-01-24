import torch
import os
import logging
import sys
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

x = np.random.normal(size=(100))
ks = 3
y = np.zeros_like(x)
for i in range(len(x)):
    if i-int(ks/2) < 0:
        y[i] = np.mean(x[:ks])
    elif i+int(ks/2) > (len(x)-1):
        y[i] = np.mean(x[-ks:])
    else:
        y[i] = np.mean(x[i-int(ks/2):i+int(ks/2)])
print(x)
print(y)
