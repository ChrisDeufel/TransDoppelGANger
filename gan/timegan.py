"""Reimplement TimeGAN-pytorch Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: October 18th 2021
Code author: Zhiwei Zhang (bitzzw@gmail.com)

-----------------------------

model.py: Network Modules

(1) Encoder
(2) Recovery
(3) Generator
(4) Supervisor
(5) Discriminator
"""

import torch
import torch.nn as nn
import torch.nn.init as init
from gan.gan_util import init_weights


class TGEncoder(nn.Module):
    """Embedding network between original feature space to latent space.

        Args:
          - input: input time-series features. (bs, seq_len, noise_dim) = (24, ?, 6)
          - h3: (num_layers, N, H). [3, ?, 24]

        Returns:
          - H: embeddings
        """

    def __init__(self, input_size, hidden_dim=24, num_layer=3):
        super(TGEncoder, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layer, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)

    def forward(self, input, sigmoid=True):
        e_outputs, _ = self.rnn(input)
        H = self.fc(e_outputs)
        if sigmoid:
            H = self.sigmoid(H)
        return H


class TGRecovery(nn.Module):
    """Recovery network from latent space to original space.

    Args:
      - H: latent representation
      - T: input time information

    Returns:
      - X_tilde: recovered data
    """

    def __init__(self, output_size, hidden_dim=24, num_layer=3):
        super(TGRecovery, self).__init__()
        # self.rnn = nn.GRU(input_size=hidden_dim, hidden_size=output_size, num_layers=num_layer)
        self.rnn = nn.LSTM(input_size=hidden_dim, hidden_size=output_size, num_layers=num_layer, batch_first=True)
        self.fc = nn.Linear(output_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)

    def forward(self, input, sigmoid=True):
        r_outputs, _ = self.rnn(input)
        X_tilde = self.fc(r_outputs)
        if sigmoid:
            X_tilde = self.sigmoid(X_tilde)
        return X_tilde


class TGGenerator(nn.Module):
    """Generator function: Generate time-series data in latent space.

    Args:
      - Z: random variables
      - T: input time information

    Returns:
      - E: generated embedding
    """

    def __init__(self, z_dim=6, hidden_dim=24, num_layer=3):
        super(TGGenerator, self).__init__()
        # self.rnn = nn.GRU(input_size=z_dim, hidden_size=hidden_dim, num_layers=num_layer)
        self.rnn = nn.LSTM(input_size=z_dim, hidden_size=hidden_dim, num_layers=num_layer, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)

    def forward(self, input, sigmoid=True):
        g_outputs, _ = self.rnn(input)
        E = self.fc(g_outputs)
        if sigmoid:
            E = self.sigmoid(E)
        return E


class TGSupervisor(nn.Module):
    """Generate next sequence using the previous sequence.

    Args:
      - H: latent representation
      - T: input time information

    Returns:
      - S: generated sequence based on the latent representations generated by the generator
    """

    def __init__(self, hidden_dim=24, num_layer=3):
        super(TGSupervisor, self).__init__()
        # self.rnn = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layer)
        self.rnn = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layer, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)

    def forward(self, input, sigmoid=True):
        s_outputs, _ = self.rnn(input)
        S = self.fc(s_outputs)
        if sigmoid:
            S = self.sigmoid(S)
        return S


class TGDiscriminator(nn.Module):
    """Discriminate the original and synthetic time-series data.

    Args:
      - H: latent representation
      - T: input time information

    Returns:
      - Y_hat: classification results between original and synthetic time-series
    """

    def __init__(self, hidden_dim=24, num_layer=3):
        super(TGDiscriminator, self).__init__()
        # self.rnn = nn.GRU(input_size=hidden_dim, hidden_size=int(hidden_dim/2), num_layers=num_layer, bidirectional=True)
        self.rnn = nn.LSTM(input_size=hidden_dim, hidden_size=int(hidden_dim / 2), num_layers=num_layer,
                          bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)

    def forward(self, input, sigmoid=True):
        d_outputs, _ = self.rnn(input)
        Y_hat = self.fc(d_outputs)
        if sigmoid:
            Y_hat = self.sigmoid(Y_hat)
        return Y_hat
