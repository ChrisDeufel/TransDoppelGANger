import torch.nn as nn
from gan.gan_util import init_weights
import torch.nn.functional as F
import torch

"""
NOTE: As the exact model architecture is not specified in https://arxiv.org/abs/1706.02633, I did implement a regular 
GAN as in 
https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/1.%20SimpleGAN/fc_gan.py 
and replaced the Generator and Discriminator with LSTMs (this is all the information the paper gives)
"""


class RGANGenerator(nn.Module):
    def __init__(self, sequence_length, output_size, device, hidden_size=None, noise_size=5, num_layers=1,
                 rnn_nonlinearity='tanh', **kwargs):
        """Recursive GAN (Generator) implementation with RNN cells.

        Layers:
            RNN (with activation, multiple layers):
                input:  (batch_size, sequence_length, noise_size)
                output: (batch_size, sequence_length, hidden_size)

            Linear (no activation, weights shared between time steps):
                input:  (batch_size, sequence_length, hidden_size)
                output: (batch_size, sequence_length, output_size)

        Args:
            sequence_length (int): Number of points in the sequence.
                Defined by the real dataset.
            output_size (int): Size of output (usually the last tensor dimension).
                Defined by the real dataset.
            hidden_size (int, optional): Size of RNN output.
                Defaults to output_size.
            noise_size (int, optional): Size of noise used to generate fake data.
                Defaults to output_size.
            num_layers (int, optional): Number of stacked RNNs in rnn.
            rnn_nonlinearity (str, optional): Non-linearity of the RNN. Must be
                either 'tanh' or 'relu'. Only valid if rnn_type == 'rnn'.
        """
        # Defaults
        super().__init__()
        noise_size = noise_size or output_size
        hidden_size = hidden_size or output_size
        self.device = device
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.noise_size = noise_size
        self.num_layers = num_layers
        self.rnn_nonlinearity = rnn_nonlinearity
        self.feature_num_layers = num_layers
        self.feature_num_units = hidden_size
        # Build RNN layer
        self.rnn = nn.LSTM(input_size=noise_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        # Initialize all weights.
        self.rnn.apply(init_weights)
        self.linear.apply(init_weights)

    def forward(self, z, sigmoid=True):
        # initial hidden and cell state
        h_o = torch.randn((self.feature_num_layers, z.size(0), self.hidden_size)).to(self.device)
        c_0 = torch.randn((self.feature_num_layers, z.size(0), self.hidden_size)).to(self.device)
        y, _ = self.rnn(z, (h_o, c_0))
        y = self.linear(y)
        if sigmoid:
            y = self.sigmoid(y)
        return y


class RGANDiscriminator(nn.Module):
    def __init__(self, sequence_length, input_size, device, hidden_size=None, num_layers=1, rnn_nonlinearity='tanh', **kwargs):
        """Recursive GAN (Discriminator) implementation with RNN cells.

        Layers:
            RNN (with activation, multiple layers):
                input:  (batch_size, sequence_length, input_size)
                output: (batch_size, sequence_length, hidden_size)

            Linear (no activation, weights shared between time steps):
                input:  (batch_size, sequence_length, hidden_size)
                output: (batch_size, sequence_length, 1)

        Args:
            sequence_length (int): Number of points in the sequence.
            input_size (int): Size of input (usually the last tensor dimension).
            hidden_size (int, optional): Size of hidden layers in rnn.
                If None, defaults to input_size.
            num_layers (int, optional): Number of stacked RNNs in rnn.
            rnn_nonlinearity (str, optional): Non-linearity of the RNN. Must be
                either 'tanh' or 'relu'. Only valid if rnn_type == 'rnn'.
        """
        # Set hidden_size to input_size if not specified
        super().__init__()
        hidden_size = hidden_size or input_size
        self.device = device
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.feature_num_units = hidden_size
        # Build RNN layer
        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

        # Initialize all weights.
        self.rnn.apply(init_weights)
        self.linear.apply(init_weights)

    def forward(self, x):
        # initial hidden and cell state
        h_o = torch.randn((self.num_layers, x.size(0), self.hidden_size)).to(self.device)
        c_0 = torch.randn((self.num_layers, x.size(0), self.hidden_size)).to(self.device)
        y, _ = self.rnn(x, (h_o, c_0))
        y = self.linear(y)
        y = torch.sigmoid(y)
        return y


class RCGANGenerator2(nn.Module):
    def __init__(self, input_feature_shape, input_attribute_shape, noise_dim=30, num_units=200, num_layers=1, alpha=0.1,
                 **kwargs):
        # Defaults
        super().__init__()
        self.input_feature_shape = input_feature_shape
        self.input_attribute_shape = input_attribute_shape
        self.noise_dim = noise_dim
        self.num_units = num_units
        self.num_layers = num_layers
        self.device = "cuda"
        # Build FC layer
        self.input_size = self.input_attribute_shape[1] + self.noise_dim
        self.output_size = self.input_feature_shape[1] * self.input_feature_shape[2]
        modules = [nn.Linear(self.input_size, self.output_size), nn.LeakyReLU(negative_slope=alpha)]
        self.gen_1 = nn.Sequential(*modules)
        # for i in range(num_layers - 2):
        # modules.append(nn.Linear(num_units, num_units))
        # modules.append(nn.LeakyReLU(negative_slope=alpha))
        self.rnn = nn.LSTM(input_size=self.input_feature_shape[2],
                           hidden_size=100,
                           num_layers=num_layers,
                           batch_first=True)
        modules = []
        modules.append(nn.Linear(self.input_feature_shape[1]*100, self.output_size))
        modules.append(nn.LeakyReLU(negative_slope=alpha))
        self.gen_2 = nn.Sequential(*modules)

        # Initialize all weights.
        self.gen_1.apply(init_weights)
        self.rnn.apply(init_weights)
        self.gen_2.apply(init_weights)

    def forward(self, x):
        x = self.gen_1(x)
        x = torch.reshape(x, (x.shape[0], self.input_feature_shape[1], self.input_feature_shape[2]))
        h_o = torch.randn((self.num_layers, 100, 100)).to(self.device)
        c_0 = torch.randn((self.num_layers, 100, 100)).to(self.device)
        x, _ = self.rnn(x, (h_o, c_0))
        x = torch.reshape(x, (x.shape[0], self.input_feature_shape[1]*100))
        x = self.gen_2(x)
        return x


class RCGANDiscriminator2(nn.Module):
    def __init__(self, input_feature_shape, input_attribute_shape, num_units=100, num_layers=3,
                 alpha=0.1, **kwargs):
        # Defaults
        super().__init__()
        self.input_feature_shape = input_feature_shape
        self.input_attribute_shape = input_attribute_shape
        self.num_units = num_units
        self.num_layers = num_layers

        # Build FC layer
        self.input_size = self.input_attribute_shape[1] + (self.input_feature_shape[1] * self.input_feature_shape[2])
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
