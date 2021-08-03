import torch
import os
import logging
import sys
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.nn.functional as F
import torch.nn as nn



print(torch.cuda.is_available())
print(torch.version.cuda)