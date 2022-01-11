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
    def __init__(self, sequence_length, output_size, hidden_size=None, noise_size=5, num_layers=1,
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

        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.noise_size = noise_size
        self.num_layers = num_layers
        self.rnn_nonlinearity = rnn_nonlinearity


        # Build RNN layer
        self.rnn = nn.LSTM(input_size=noise_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers)
        self.linear = nn.Linear(hidden_size, output_size)
        # Initialize all weights.
        self.rnn.apply(init_weights)
        self.linear.apply(init_weights)

    def forward(self, z):
        y, _ = self.rnn(z)
        y = self.linear(y)
        return y


class RGANDiscriminator(nn.Module):
    def __init__(self, sequence_length, input_size, hidden_size=None, num_layers=1, rnn_nonlinearity='tanh', **kwargs):
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

        self.input_size = input_size
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Build RNN layer
        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers)
        self.linear = nn.Linear(hidden_size, 1)

        # Initialize all weights.
        self.rnn.apply(init_weights)
        self.linear.apply(init_weights)

    def forward(self, x):
        y, _ = self.rnn(x)
        y = self.linear(y)
        y = torch.sigmoid(y)
        return y
