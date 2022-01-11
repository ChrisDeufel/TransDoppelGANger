"""Reimplement TimeGAN-pytorch Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: October 18th 2021
Code author: Zhiwei Zhang (bitzzw@gmail.com)

-----------------------------

timegan.py

Note: Use original data as training set to generater synthetic data (time-series)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data import TimeGanData
from util import extract_len, random_generator, NormMinMax, setup_logger, add_handler
from gan.timegan import Encoder, Recovery, Generator, Discriminator, Supervisor
from torch.utils.data import DataLoader
import logging

# time_logger = logging.getLogger(__name__)
# time_logger.setLevel(logging.INFO)
# config_logger = logging.getLogger(__name__)
# config_logger.setLevel(logging.INFO)


# def add_handler_trainer(handlers):
#     for handler in handlers:
#         time_logger.addHandler(handler)
#
#
# def add_config_handler_trainer(handlers):
#     for handler in handlers:
#         config_logger.addHandler(handler)


class TimeGAN:
    """TimeGAN Class
    """

    @property
    def name(self):
        return 'TimeGAN'

    def __init__(self,
                 train_loader,
                 device,
                 lr=0.001,
                 z_dim=6,
                 hidden_dim=24,
                 num_layer=3,
                 beta1=0.5,
                 istrain=True,
                 w_gamma=1,
                 w_es=0.1,
                 w_e0=10,
                 w_g=100,
                 checkpoint_dir="",
                 config_log="",
                 time_log=""):
        torch.backends.cudnn.enabled = False
        # Initalize variables.
        self.train_dl = train_loader
        self.device = device
        self.lr = lr
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.beta1 = beta1
        self.isTrain = istrain
        self.w_gamma = w_gamma
        self.w_es = w_es
        self.w_e0 = w_e0
        self.w_g = w_g
        self.checkpoint_dir = checkpoint_dir
        if config_log is not None:
            self.config_logger = setup_logger(name="config_logger", log_file=config_log,
                                              formatter=logging.Formatter('%(message)s'))
            self.time_logger = setup_logger(name="time_logger", log_file=time_log,
                                            formatter=logging.Formatter('%(asctime)s:%(message)s'))
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.INFO)
            add_handler(handlers=[stream_handler], logger=self.time_logger)
            self.config_logger.info("Learning Rate: {0}".format(self.lr))
            self.config_logger.info("Z-Dimension: {0}".format(self.z_dim))
            self.config_logger.info("Hidden Dimension: {0}".format(self.hidden_dim))
            self.config_logger.info("Num Layer: {0}".format(self.num_layer))
            self.config_logger.info("Beta 1: {0}".format(self.beta1))
            self.config_logger.info("w_gamma: {0}".format(self.w_gamma))
            self.config_logger.info("w_es: {0}".format(self.w_es))
            self.config_logger.info("w_g: {0}".format(self.w_g))
            self.config_logger.info("w_e0: {0}".format(self.w_e0))
            self.config_logger.info("data_feature_shape: {0}".format(self.train_dl.dataset.data_feature_shape))

        # calculate variables
        self.min_val = train_loader.dataset.min_val
        self.max_val = train_loader.dataset.max_val
        self.max_seq_len = train_loader.dataset.max_seq_len
        self.lengths = train_loader.dataset.lengths
        self.data_num, _, _ = train_loader.dataset.data_feature.shape  # 3661; 24; 6

        # -- Misc attributes
        # Create and initialize networks.
        # determine input size for encoder
        num_features = self.train_dl.dataset.data_feature.shape[2]
        self.nete = Encoder(input_size=num_features, hidden_dim=self.hidden_dim, num_layer=self.num_layer).to(
            self.device)
        self.netr = Recovery(output_size=num_features, hidden_dim=self.hidden_dim, num_layer=self.num_layer).to(
            self.device)
        self.netg = Generator(z_dim=self.z_dim, hidden_dim=self.hidden_dim, num_layer=self.num_layer).to(self.device)
        self.netd = Discriminator(z_dim=self.z_dim, hidden_dim=self.hidden_dim, num_layer=self.num_layer).to(
            self.device)
        self.nets = Supervisor(z_dim=self.z_dim, hidden_dim=self.hidden_dim, num_layer=self.num_layer).to(self.device)
        if config_log is not None:
            self.config_logger.info("nete: {0}".format(self.nete))
            self.config_logger.info("netr: {0}".format(self.netr))
            self.config_logger.info("netg: {0}".format(self.netg))
            self.config_logger.info("netd: {0}".format(self.netd))
            self.config_logger.info("nets: {0}".format(self.nets))

        # loss
        self.l_mse = nn.MSELoss()
        self.l_r = nn.L1Loss()
        self.l_bce = nn.BCELoss()
        # Setup optimizer
        if self.isTrain:
            self.nete.train()
            self.netr.train()
            self.netg.train()
            self.netd.train()
            self.nets.train()
            self.optimizer_e = optim.Adam(self.netd.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
            self.optimizer_r = optim.Adam(self.netg.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
            self.optimizer_g = optim.Adam(self.netd.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
            self.optimizer_d = optim.Adam(self.netg.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
            self.optimizer_s = optim.Adam(self.netd.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

    def save(self, epoch):
        if not os.path.exists("{0}/epoch_{1}".format(self.checkpoint_dir, epoch)):
            os.makedirs("{0}/epoch_{1}".format(self.checkpoint_dir, epoch))
        torch.save(self.nete, "{0}/epoch_{1}/nete.pth".format(self.checkpoint_dir, epoch))
        torch.save(self.netr, "{0}/epoch_{1}/netr.pth".format(self.checkpoint_dir, epoch))
        torch.save(self.netg, "{0}/epoch_{1}/netg.pth".format(self.checkpoint_dir, epoch))
        torch.save(self.netd, "{0}/epoch_{1}/netd.pth".format(self.checkpoint_dir, epoch))
        torch.save(self.nets, "{0}/epoch_{1}/nets.pth".format(self.checkpoint_dir, epoch))

    def load(self, model_dir=None):
        if not os.path.exists(model_dir):
            raise Exception("Directory to load pytorch model doesn't exist")
        self.nete = torch.load("{0}/nete.pth".format(model_dir))
        self.netr = torch.load("{0}/netr.pth".format(model_dir))
        self.netg = torch.load("{0}/netg.pth".format(model_dir))
        self.netd = torch.load("{0}/netd.pth".format(model_dir))
        self.nets = torch.load("{0}/nets.pth".format(model_dir))
        self.nete = self.nete.to(self.device)
        self.netr = self.netr.to(self.device)
        self.netg = self.netg.to(self.device)
        self.netd = self.netd.to(self.device)
        self.nets = self.nets.to(self.device)

    def forward_e(self):
        """ Forward propagate through netE
        """
        self.H = self.nete(self.X)

    def forward_er(self):
        """ Forward propagate through netR"""
        self.H = self.nete(self.X)
        self.X_tilde = self.netr(self.H)

    def forward_g(self):
        """ Forward propagate through netG
        """
        self.E_hat = self.netg(self.Z)

    def forward_dg(self):
        """ Forward propagate through netD
        """
        self.Y_fake = self.netd(self.H_hat)
        self.Y_fake_e = self.netd(self.E_hat)

    def forward_rg(self):
        """ Forward propagate through netG
        """
        self.X_hat = self.netr(self.H_hat)

    def forward_s(self):
        """ Forward propagate through netS
        """
        self.H_supervise = self.nets(self.H)

    def forward_sg(self):
        """ Forward propagate through netS
        """
        self.H_hat = self.nets(self.E_hat)

    def forward_d(self):
        """ Forward propagate through netD
        """
        self.Y_real = self.netd(self.H)
        self.Y_fake = self.netd(self.H_hat)
        self.Y_fake_e = self.netd(self.E_hat)

    def backward_er(self):
        """ Backpropagate through netE
        """
        self.err_er = self.l_mse(self.X, self.X_tilde)
        self.err_er.backward(retain_graph=True)

        # print("Loss: ", self.err_er)

    def backward_g(self):
        """ Backpropagate through netG
        """
        self.err_g_U = self.l_bce(torch.ones_like(self.Y_fake), self.Y_fake)
        self.err_g_U_e = self.l_bce(torch.ones_like(self.Y_fake_e), self.Y_fake_e)
        self.err_g_V1 = torch.mean(torch.abs(torch.sqrt(torch.std(self.X_hat, [0])[1] + 1e-6) - torch.sqrt(
            torch.std(self.X, [0])[1] + 1e-6)))  # |a^2 - b^2|
        self.err_g_V2 = torch.mean(
            torch.abs((torch.mean(self.X_hat, [0])[0]) - (torch.mean(self.X, [0])[0])))  # |a - b|
        self.err_g = self.err_g_U + \
                     self.err_g_U_e * self.w_gamma + \
                     self.err_g_V1 * self.w_g + \
                     self.err_g_V2 * self.w_g

        self.err_g.backward(retain_graph=True)

    def backward_s(self):
        """ Backpropagate through netS
        """
        self.err_s = self.l_mse(self.H[:, 1:, :], self.H_supervise[:, :-1, :])
        self.err_s.backward(retain_graph=True)

    def backward_d(self):
        """ Backpropagate through netD
        """
        self.err_d_real = self.l_bce(torch.ones_like(self.Y_real), self.Y_real)
        self.err_d_fake = self.l_bce(torch.ones_like(self.Y_fake), self.Y_fake)
        self.err_d_fake_e = self.l_bce(torch.ones_like(self.Y_fake_e), self.Y_fake_e)
        self.err_d = self.err_d_real + \
                     self.err_d_fake + \
                     self.err_d_fake_e * self.w_gamma
        if self.err_d > 0.15:
            self.err_d.backward(retain_graph=True)

    def optimize_params_er(self):
        """ Forwardpass, Loss Computation and Backwardpass.
        """
        # Forward-pass
        # Forward-pass
        self.forward_er()

        # Backward-pass
        # nete & netr
        self.optimizer_e.zero_grad()
        self.optimizer_r.zero_grad()
        self.backward_er()
        self.optimizer_e.step()
        self.optimizer_r.step()

    def optimize_params_s(self):
        """ Forwardpass, Loss Computation and Backwardpass.
        """
        # Forward-pass
        self.forward_e()
        self.forward_s()

        # Backward-pass
        # nets
        self.optimizer_s.zero_grad()
        self.backward_s()
        self.optimizer_s.step()

    def optimize_params_g(self):
        """ Forwardpass, Loss Computation and Backwardpass.
        """
        # Forward-pass
        self.forward_g()
        self.forward_sg()
        self.forward_rg()
        self.forward_dg()

        # Backward-pass
        # nets
        self.optimizer_g.zero_grad()
        self.backward_g()
        self.optimizer_g.step()

    def optimize_params_d(self):
        """ Forwardpass, Loss Computation and Backwardpass.
        """
        # Forward-pass
        self.forward_e()
        self.forward_g()
        self.forward_sg()
        # self.forward_dg()
        self.forward_d()

        # Backward-pass
        # nets
        self.optimizer_d.zero_grad()
        self.backward_d()
        self.optimizer_d.step()

    def seed(self, seed_value):
        """ Seed

        Arguments:
            seed_value {int} -- [description]
        """
        # Check if seed is default value
        if seed_value == -1:
            return

        # Otherwise seed all functionality
        import random
        random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        np.random.seed(seed_value)
        torch.backends.cudnn.deterministic = True

    def save_weights(self, epoch):
        """Save net weights for the current epoch.

        Args:
            epoch ([int]): Current epoch number.
        """

        weight_dir = os.path.join(self.opt.outf, self.opt.name, 'train', 'weights')
        if not os.path.exists(weight_dir): os.makedirs(weight_dir)

        torch.save({'epoch': epoch + 1, 'state_dict': self.nete.state_dict()},
                   '%s/netE.pth' % (weight_dir))
        torch.save({'epoch': epoch + 1, 'state_dict': self.netr.state_dict()},
                   '%s/netR.pth' % (weight_dir))
        torch.save({'epoch': epoch + 1, 'state_dict': self.netg.state_dict()},
                   '%s/netG.pth' % (weight_dir))
        torch.save({'epoch': epoch + 1, 'state_dict': self.netd.state_dict()},
                   '%s/netD.pth' % (weight_dir))
        torch.save({'epoch': epoch + 1, 'state_dict': self.nets.state_dict()},
                   '%s/netS.pth' % (weight_dir))

    def train_one_iter_er(self, lengths, data_feature):
        """ Train the model for one epoch.
        """
        self.nete.train()
        self.netr.train()

        self.X, self.T = data_feature, lengths

        # train encoder & decoder
        self.optimize_params_er()

    def train_one_iter_s(self, lengths, data_feature):
        """ Train the model for one epoch.
        """

        self.nete.eval()
        self.nets.train()

        # set mini-batch
        self.X, self.T = data_feature, lengths

        # train superviser
        self.optimize_params_s()

    def train_one_iter_g(self, lengths, data_feature):
        """ Train the model for one epoch.
        """
        self.batch_size = data_feature.shape[0]
        self.netr.eval()
        self.nets.eval()
        self.netd.eval()
        self.netg.train()

        # set mini-batch
        self.X, self.T = data_feature, lengths
        self.Z = torch.tensor(random_generator(self.batch_size, self.z_dim, self.T, self.max_seq_len),
                              dtype=torch.float32).to(self.device)

        # train superviser
        self.optimize_params_g()

    def train_one_iter_d(self, lengths, data_feature):
        """ Train the model for one epoch.
        """
        self.nete.eval()
        self.netr.eval()
        self.nets.eval()
        self.netg.eval()
        self.netd.train()

        # set mini-batch
        self.X, self.T = data_feature, lengths
        self.Z = torch.tensor(random_generator(self.batch_size, self.z_dim, self.T, self.max_seq_len),
                              dtype=torch.float32).to(self.device)

        # train superviser
        self.optimize_params_d()

    def train(self, epochs, writer_frequency=1, saver_frequency=10):
        """ Train the model
        """
        for iter in range(epochs):
            for batch_idx, (lengths, data_feature) in enumerate(self.train_dl):
                lengths = lengths.to(self.device)
                data_feature = data_feature.to(self.device)
                self.train_one_iter_er(lengths, data_feature)
            self.time_logger.info('END OF EPOCH {0} - ENCODER/RECOVERY'.format(iter))

        for iter in range(epochs):
            for batch_idx, (lengths, data_feature) in enumerate(self.train_dl):
                lengths = lengths.to(self.device)
                data_feature = data_feature.to(self.device)
                self.train_one_iter_s(lengths, data_feature)
            self.time_logger.info('END OF EPOCH {0} - GENERATOR'.format(iter))

        for iter in range(epochs):
            for batch_idx, (lengths, data_feature) in enumerate(self.train_dl):
                lengths = lengths.to(self.device)
                data_feature = data_feature.to(self.device)
                for kk in range(2):
                    self.train_one_iter_g(lengths, data_feature)
                    self.train_one_iter_er(lengths, data_feature)
                self.train_one_iter_d(lengths, data_feature)
            self.time_logger.info('END OF EPOCH {0} - GENERATOR/DISCRIMINATOR'.format(iter))
            if iter % saver_frequency == 0:
                self.save(iter)

    def sample_from(self, batch_size, return_gen_flag_feature=False):
        ## Synthetic data generation
        self.Z = torch.tensor(random_generator(batch_size, self.z_dim, self.lengths, self.max_seq_len),
                              dtype=torch.float32).to(self.device)
        self.E_hat = self.netg(self.Z)  # [?, 24, 24]
        self.H_hat = self.nets(self.E_hat)  # [?, 24, 24]
        features = self.netr(self.H_hat)  # [?, 24, 24]
        features = features.detach().cpu().numpy()
        attributes = torch.zeros((batch_size, self.train_dl.dataset.data_attribute_shape[1]))
        gen_flags = np.zeros(features.shape[:-1])
        lengths = np.zeros(features.shape[0])
        for i in range(len(features)):
            winner = (features[i, :, -1] > features[i, :, -2])
            argmax = np.argmax(winner == True)
            if argmax == 0:
                gen_flags[i, :] = 1
            else:
                gen_flags[i, :argmax] = 1
            lengths[i] = argmax
        if not return_gen_flag_feature:
            features = features[:, :, :-2]
        return features, attributes.cpu().numpy(), gen_flags, lengths
