import torch.nn as nn
from gan.gan_util import init_weights
import torch.nn.functional as F
import torch
"""
Implementation like in https://arxiv.org/abs/1406.2661
"""

class NaiveGanGenerator(nn.Module):
    def __init__(self, input_feature_shape, input_attribute_shape, noise_dim=30, num_units=200, num_layers=3, alpha=0.1,
                 **kwargs):
        # Defaults
        super().__init__()
        self.input_feature_shape = input_feature_shape
        self.input_attribute_shape = input_attribute_shape
        self.noise_dim = noise_dim
        self.num_units = num_units
        self.num_layers = num_layers

        # Build FC layer
        self.input_size = self.noise_dim
        self.output_size = self.input_feature_shape[1]*self.input_feature_shape[2]
        modules = [nn.Linear(self.input_size, num_units), nn.LeakyReLU(negative_slope=alpha)]
        for i in range(num_layers - 2):
            modules.append(nn.Linear(num_units, num_units))
            modules.append(nn.LeakyReLU(negative_slope=alpha))
        modules.append(nn.Linear(num_units, self.output_size))
        modules.append(nn.LeakyReLU(negative_slope=alpha))
        self.gen = nn.Sequential(*modules)
        # Initialize all weights.
        self.gen.apply(init_weights)

    def forward(self, x):
        return self.gen(x)


class NaiveGanDiscriminator(nn.Module):
    def __init__(self, input_feature_shape, input_attribute_shape, num_units=200, num_layers=3,
                 alpha=0.1, **kwargs):
        # Defaults
        super().__init__()
        self.input_feature_shape = input_feature_shape
        self.input_attribute_shape = input_attribute_shape
        self.num_units = num_units
        self.num_layers = num_layers

        # Build FC layer
        self.input_size = self.input_feature_shape[1] * self.input_feature_shape[2]
        modules = [nn.Linear(self.input_size, num_units), nn.LeakyReLU(negative_slope=alpha)]
        for i in range(num_layers - 2):
            modules.append(nn.Linear(num_units, num_units))
            modules.append(nn.LeakyReLU(negative_slope=alpha))
        modules.append(nn.Linear(num_units, 1))
        # modules.append(nn.LeakyReLU(negative_slope=alpha))
        self.disc = nn.Sequential(*modules)
        # Initialize all weights.
        self.disc.apply(init_weights)

    def forward(self, x):
        x = self.disc(x)
        return torch.sigmoid(x)