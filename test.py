import torch
import os
import logging
import sys
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

def plot_seq_len(data_bins):
    fig, axes = plt.subplots(1, 1, figsize=(12, 4))
    x = np.arange(start=0, stop=len(data_bins), step=1)
    axes.bar(x, data_bins, color="blue", alpha=0.5)
    axes.legend()
    axes.set_title("Sequence Length")
    plt.savefig('{0}.png'.format("test"))


def sequence_length(data_gen_flag):
    """
    :param dataset: dataset name
    :param data: List of dictionaries (one dictionary per features with keys 'data_gen_flag', 'name' and 'color')
    :return: pass
    """
    len = np.count_nonzero(data_gen_flag, axis=1)
    data_bins = np.bincount(len)[:50]
    plot_seq_len(data_bins)


#data_path = "runs/google/test/1/checkpoint/generated_samples.npz"
data_path = "data/google/data_train.npz"
data = np.load(data_path)
#gen_flags = data['sampled_gen_flags']
#features = data['sampled_features']
#features = data['data_feature']
gen_flags = data['data_gen_flag']
sequence_length(gen_flags)
# gen_flags = np.zeros(features.shape[:-1])
# for i in range(len(features)):
#     winner = (features[i, :, -1] > features[i, :, -2])
#     argmax = np.argmax(winner==True)
#     gen_flags[i, :argmax] = 1
# sequence_length(gen_flags)
# data = np.load("data/transactions/data_train.npz")
# attributes = data['data_attribute']
# for u in range(len(attributes)):
#     np.save("data/transactions/{}_data_attribute.npy".format(u), attributes[u])
