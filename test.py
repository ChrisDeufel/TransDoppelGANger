import torch
import os
import logging
import sys
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

feature_open = np.load("data/index_course/data_feature_n_g.npy")
attribute_open = np.load("data/index_course/data_attribute_n_g.npy")

feature_growth = np.load("data/index_growth/data_feature_n_g.npy")
attribute_growth = np.load("data/index_growth/data_attribute_n_g.npy")

print('hello')
