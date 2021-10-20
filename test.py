import torch
import os
import logging
import sys
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

data = np.load("data/transactions/data_train.npz")
attributes = data['data_attribute']
for u in range(len(attributes)):
    np.save("data/transactions/{}_data_attribute.npy".format(u), attributes[u])
