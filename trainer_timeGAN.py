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
from gan.timegan import Encoder, Recovery, Generator, Discriminator, Supervisor
from gan.gan_util import gen_noise
import logging


def add_handler(logger, handlers):
    for handler in handlers:
        logger.addHandler(handler)


def setup_logging(time_logging_file, config_logging_file):
    # SET UP LOGGING
    config_logger = logging.getLogger("config_logger")
    config_logger.setLevel(logging.INFO)
    # config_logger.setLevel(logging.INFO)
    time_logger = logging.getLogger("time_logger")
    time_logger.setLevel(logging.INFO)
    # time_logger.setLevel(logging.INFO)
    # set up time handler
    time_formatter = logging.Formatter('%(asctime)s:%(message)s')
    time_handler = logging.FileHandler(time_logging_file)
    time_handler.setLevel(logging.INFO)
    time_handler.setFormatter(time_formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(time_formatter)
    add_handler(time_logger, [time_handler, stream_handler])
    # setup config handler
    config_formatter = logging.Formatter('%(message)s')
    config_handler = logging.FileHandler(config_logging_file)
    config_handler.setLevel(logging.INFO)
    config_handler.setFormatter(config_formatter)
    config_logger.addHandler(config_handler)
    return config_logger, time_logger


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
                 w_gamma=1,
                 w_es=0.1,
                 w_e0=10,
                 w_g=100,
                 checkpoint_dir="",
                 config_logging_file="",
                 time_logging_file=""):
        self.config_logger, self.time_logger = setup_logging(time_logging_file, config_logging_file)
        # Initalize variables.
        self.real_train_dl = train_loader
        self.device = device
        self.lr = lr
        self.z_dim = self.real_train_dl.dataset.data_feature_shape[2]
        self.hidden_dim = 4 * self.real_train_dl.dataset.data_feature_shape[2]
        self.num_layer = num_layer
        self.beta1 = beta1
        self.w_gamma = w_gamma
        self.w_es = w_es
        self.w_e0 = w_e0
        self.w_g = w_g
        self.checkpoint_dir = checkpoint_dir

        # Create and initialize networks.
        # determine input size for encoder
        num_features = self.real_train_dl.dataset.data_feature.shape[2]
        self.nete = Encoder(input_size=num_features, hidden_dim=self.hidden_dim, num_layer=self.num_layer).to(
            self.device)
        self.netr = Recovery(output_size=num_features, hidden_dim=self.hidden_dim, num_layer=self.num_layer).to(
            self.device)
        self.netg = Generator(z_dim=self.z_dim, hidden_dim=self.hidden_dim, num_layer=self.num_layer).to(self.device)
        self.netd = Discriminator(hidden_dim=self.hidden_dim, num_layer=self.num_layer).to(
            self.device)
        self.nets = Supervisor(hidden_dim=self.hidden_dim, num_layer=self.num_layer).to(self.device)
        # loss
        self.l_mse = nn.MSELoss()
        self.l_r = nn.L1Loss()
        self.l_bce = nn.BCELoss()
        # Setup optimizer
        self.optimizer_er = optim.Adam(list(self.nete.parameters()) + list(self.netr.parameters()), lr=self.lr,
                                       betas=(self.beta1, 0.999))
        self.optimizer_d = optim.Adam(self.netd.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimizer_gs = optim.Adam(list(self.netg.parameters()) + list(self.nets.parameters()), lr=self.lr,
                                       betas=(self.beta1, 0.999))

        self.config_logger.info("EMBEDDER: {0}".format(self.nete))
        self.config_logger.info("RECOVERY: {0}".format(self.netr))
        self.config_logger.info("DISCRIMINATOR: {0}".format(self.netd))
        self.config_logger.info("GENERATOR: {0}".format(self.netg))
        self.config_logger.info("SUPERVISOR: {0}".format(self.nets))

        self.config_logger.info("EMBEDDER OPTIMIZER: {0}".format(self.optimizer_er))
        self.config_logger.info("RECOVERY OPTIMIZER: {0}".format(self.optimizer_r))
        self.config_logger.info("DISCRIMINATOR OPTIMIZER: {0}".format(self.optimizer_d))
        self.config_logger.info("GENERATOR OPTIMIZER: {0}".format(self.optimizer_gs))
        self.config_logger.info("SUPERVISOR OPTIMIZER: {0}".format(self.optimizer_s))

        self.config_logger.info("Reconstruction Loss: {0}".format(self.l_mse))
        self.config_logger.info("Unsupervised Loss: {0}".format(self.l_mse))
        self.config_logger.info("Generator Adversarial Loss: {0}".format(self.l_bce))

        self.config_logger.info("Discriminator Adversarial Loss: {0}".format(self.l_bce))

        self.config_logger.info("Z-Dimension: {0}".format(self.z_dim))
        self.config_logger.info("Hidden Dimension: {0}".format(self.hidden_dim))
        self.config_logger.info("Beta 1: {0}".format(self.beta1))
        self.config_logger.info("w_gamma: {0}".format(self.w_gamma))
        self.config_logger.info("w_es: {0}".format(self.w_es))
        self.config_logger.info("w_g: {0}".format(self.w_g))
        self.config_logger.info("w_e0: {0}".format(self.w_e0))

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

    def inference(self, epoch, model_dir=None):
        if model_dir is None:
            model_dir = "{0}/epoch_{1}".format(self.checkpoint_dir, epoch)
        batch_size = self.batch_size

        while self.real_train_dl.dataset.data_attribute_shape[0] % batch_size != 0:
            batch_size -= 1
        rounds = self.real_train_dl.dataset.data_attribute_shape[0] // batch_size
        sampled_features = np.zeros((0, self.real_train_dl.dataset.data_feature_shape[1],
                                     self.real_train_dl.dataset.data_feature_shape[2] - 2))
        sampled_attributes = np.zeros((0, self.real_train_dl.dataset.data_attribute_shape[1]))
        sampled_gen_flags = np.zeros((0, self.real_train_dl.dataset.data_feature_shape[1]))
        sampled_lengths = np.zeros(0)
        for i in range(rounds):
            features, attributes, gen_flags, lengths = self.sample_from(batch_size=batch_size)
            sampled_features = np.concatenate((sampled_features, features), axis=0)
            sampled_attributes = np.concatenate((sampled_attributes, attributes), axis=0)
            sampled_gen_flags = np.concatenate((sampled_gen_flags, gen_flags), axis=0)
            sampled_lengths = np.concatenate((sampled_lengths, lengths), axis=0)
        np.savez("{0}/generated_samples.npz".format(model_dir), sampled_features=sampled_features,
                 sampled_attributes=sampled_attributes, sampled_gen_flags=sampled_gen_flags,
                 sampled_lengths=sampled_lengths)

    def sample_from(self, batch_size, return_gen_flag_feature=False):
        ## Synthetic data generation
        Z = gen_noise((batch_size, self.real_train_dl.dataset.data_feature_shape[1], self.z_dim))
        E_hat = self.netg(Z)  # [?, 24, 24]
        H_hat = self.nets(E_hat)  # [?, 24, 24]
        features = self.netr(H_hat)  # [?, 24, 24]
        features = features.detach().cpu().numpy()
        attributes = torch.zeros((batch_size, self.real_train_dl.dataset.data_attribute_shape[1]))
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

    def train_one_iter_er(self, data_feature):
        """ Train the model for one epoch.
        """
        self.nete.train()
        self.netr.train()
        X = data_feature
        # train encoder & decoder
        # Forward-pass
        H = self.nete(X)
        X_tilde = self.netr(H)
        # Backward-pass
        # nete & netr
        self.optimizer_er.zero_grad()
        # backward pass
        err_e_T0 = self.l_mse(X, X_tilde)
        err_e_0 = 10 * torch.sqrt(err_e_T0)
        err_e_0.backward(retain_graph=True)
        self.optimizer_er.step()

    def train_one_iter_er_2(self, data_feature):
        """ Train the model for one epoch.
        """
        self.nete.train()
        self.netr.train()
        self.netg.eval()
        self.nets.eval()
        X = data_feature
        # Forward-pass
        H = self.nete(X)
        H_supervise = self.nets(H)
        X_tilde = self.netr(H)
        # Backward-pass
        # G loss s
        # nete & netr
        self.optimizer_er.zero_grad()
        # backward pass
        err_s = self.l_mse(H[:, 1:, :], H_supervise[:, :-1, :])
        err_e_T0 = self.l_mse(X, X_tilde)
        err_e_0 = 10 * torch.sqrt(err_e_T0)
        err_e = err_e_0 + 0.1 * err_s
        err_e.backward(retain_graph=True)
        self.optimizer_er.step()

    def train_one_iter_s(self, data_feature):
        """ Train the model for one epoch.
        """
        self.nete.eval()
        self.nets.train()
        # set mini-batch
        X = data_feature
        # Forward-pass
        H = self.nete(X)
        H_supervise = self.nets(H)
        # Backward-pass
        self.optimizer_gs.zero_grad()
        err_s = self.l_mse(H[:, 1:, :], H_supervise[:, :-1, :])
        err_s.backward(retain_graph=True)
        self.optimizer_gs.step()

    def train_one_iter_g(self, data_feature):
        """ Train the model for one epoch.
        """
        self.batch_size = data_feature.shape[0]
        self.netr.eval()
        self.nets.train()
        self.netd.eval()
        self.netg.train()

        # set mini-batch
        X = data_feature
        Z = gen_noise((self.batch_size, data_feature.shape[1], self.z_dim))
        # Forward-pass
        H = self.nete(X)
        H_supervise = self.nets(H)
        E_hat = self.netg(Z)
        H_hat = self.nets(E_hat)
        Y_fake = self.netd(H_hat)
        X_hat = self.netr(H_hat)
        Y_fake_e = self.netd(E_hat)
        # Backward-pass
        self.optimizer_gs.zero_grad()
        # G loss s
        err_s = self.l_mse(H[:, 1:, :], H_supervise[:, :-1, :])
        # G loss u
        err_g_U = self.l_bce(torch.ones_like(Y_fake), Y_fake)
        # G loss v
        err_g_V1 = torch.mean(torch.abs(torch.sqrt(torch.std(X_hat, [0])[1] + 1e-6) - torch.sqrt(
            torch.std(X, [0])[1] + 1e-6)))  # |a^2 - b^2|
        err_g_V2 = torch.mean(
            torch.abs((torch.mean(X_hat, [0])[0]) - (torch.mean(X, [0])[0])))  # |a - b|
        err_g_V = err_g_V1 + err_g_V2
        # G loss ue
        err_g_U_e = self.l_bce(torch.ones_like(Y_fake_e), Y_fake_e)
        err_g = err_g_U + self.w_gamma * err_g_U_e + 100 * torch.sqrt(err_s) + 100 * err_g_V
        err_g.backward(retain_graph=True)
        self.optimizer_gs.step()

    def train_one_iter_d(self, data_feature):
        """ Train the model for one epoch.
        """
        self.nete.eval()
        self.netr.eval()
        self.nets.eval()
        self.netg.eval()
        self.netd.train()

        # set mini-batch
        X = data_feature
        Z = gen_noise((self.batch_size, data_feature.shape[1], self.z_dim))

        # train superviser
        # forward pass
        H = self.nete(X)
        E_hat = self.netg(Z)
        H_hat = self.nets(E_hat)
        # self.forward_dg()
        Y_real = self.netd(H)
        Y_fake = self.netd(H_hat)
        Y_fake_e = self.netd(E_hat)

        # Backward-pass
        self.optimizer_d.zero_grad()
        err_d_real = self.l_bce(torch.ones_like(Y_real), Y_real)
        err_d_fake = self.l_bce(torch.ones_like(Y_fake), Y_fake)
        err_d_fake_e = self.l_bce(torch.ones_like(Y_fake_e), Y_fake_e)
        err_d = err_d_real + err_d_fake + err_d_fake_e * self.w_gamma
        if self.err_d > 0.15:
            err_d.backward(retain_graph=True)
        self.optimizer_d.step()

    def train(self, epochs, saver_frequency=10):
        """ Train the model
        """
        self.time_logger.info('Start Embedding Network Training')
        for iter in range(epochs):
            for batch_idx, (data_attribute, data_feature) in enumerate(self.real_train_dl):
                data_feature = data_feature.to(self.device)
                self.train_one_iter_er(data_feature)
            self.time_logger.info('Embedding Network Training - END OF EPOCH {0}'.format(iter))
        self.time_logger.info('Finish Embedding Network Training')

        self.time_logger.info('Start Training with Supervised Loss Only')
        for iter in range(epochs):
            for batch_idx, (data_attribute, data_feature) in enumerate(self.real_train_dl):
                data_feature = data_feature.to(self.device)
                self.train_one_iter_s(data_feature)
            self.time_logger.info('Supervised Loss Only - END OF EPOCH {0}'.format(iter))
        self.time_logger.info('Finish Training with Supervised Loss Only')

        self.time_logger.info('Start Joint Training')
        for iter in range(epochs):
            for batch_idx, (data_attribute, data_feature) in enumerate(self.real_train_dl):
                data_feature = data_feature.to(self.device)
                for kk in range(2):
                    # Train Generator and Supervisor
                    self.train_one_iter_g(data_feature)
                    # Train Embedder and Recovery again
                    self.train_one_iter_er_2(data_feature)
                self.train_one_iter_d(data_feature)
            self.time_logger.info('Joint Training - END OF EPOCH {0}'.format(iter))
            if iter % saver_frequency == 0:
                self.save(iter)
        self.time_logger.info('Finish Joint Training')