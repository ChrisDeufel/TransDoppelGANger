import torch.nn as nn


def init_weights(layer):
    if type(layer) == nn.Linear:
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)
    if type(layer) == nn.Parameter:
        nn.init.xavier_uniform_(layer)
    if type(layer) == nn.LSTM:
        for name, W in layer.named_parameters():
            if len(W.shape) > 1:
                nn.init.xavier_uniform_(W)
            else:
                nn.init.ones_(W)