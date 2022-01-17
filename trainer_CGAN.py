import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from gan.cgan import CGANGenerator, CGANDiscriminator
from gan.gan_util import gen_noise
import numpy as np
from util import calculate_mmd_rbf


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


class CGAN:

    @property
    def name(self):
        return 'CGAN'

    def __init__(self,
                 train_loader,
                 device,
                 batch_size=28,
                 lr=0.001,
                 noise_dim=30,
                 num_units_dis=100,
                 num_units_gen=200,
                 num_layers=3,
                 beta1=0.5,
                 alpha=0.1,
                 checkpoint_dir="",
                 time_logging_file='',
                 config_logging_file='',
                 isWasserstein=True):
        self.config_logger, self.time_logger = setup_logging(time_logging_file, config_logging_file)
        # setup models
        self.device = device
        self.real_train_dl = train_loader
        self.checkpoint_dir = checkpoint_dir
        self.isWasserstein = isWasserstein
        self.noise_dim = noise_dim
        self.generator = CGANGenerator(input_feature_shape=train_loader.dataset.data_feature_shape,
                                       input_attribute_shape=train_loader.dataset.data_attribute_shape,
                                       noise_dim=noise_dim, num_units=num_units_gen, num_layers=num_layers, alpha=alpha)
        self.discriminator = CGANDiscriminator(input_feature_shape=train_loader.dataset.data_feature_shape,
                                               input_attribute_shape=train_loader.dataset.data_attribute_shape,
                                               num_units=num_units_dis, num_layers=num_layers, alpha=alpha)
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)
        self.config_logger.info("DISCRIMINATOR: {0}".format(self.discriminator))
        self.config_logger.info("GENERATOR: {0}".format(self.generator))
        # loss
        self.criterion = nn.BCELoss()
        self.config_logger.info("Criterion: {0}".format(self.criterion))
        # Setup optimizer
        self.optimizer_gen = optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizer_dis = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.config_logger.info("DISCRIMINATOR OPTIMIZER: {0}".format(self.optimizer_dis))
        self.config_logger.info("GENERATOR OPTIMIZER: {0}".format(self.optimizer_gen))
        self.batch_size = batch_size
        self.config_logger.info("Batch Size: {0}".format(self.batch_size))
        self.config_logger.info("Noise Dimension: {0}".format(self.noise_dim))
        self.config_logger.info("d_rounds: {0}".format("1"))
        self.config_logger.info("g_rounds: {0}".format("1"))
        self.config_logger.info("Device: {0}".format(self.device))

    def save(self, epoch):
        if not os.path.exists("{0}/epoch_{1}".format(self.checkpoint_dir, epoch)):
            os.makedirs("{0}/epoch_{1}".format(self.checkpoint_dir, epoch))
        torch.save(self.generator, "{0}/epoch_{1}/generator.pth".format(self.checkpoint_dir, epoch))
        torch.save(self.discriminator, "{0}/epoch_{1}/discriminator.pth".format(self.checkpoint_dir, epoch))

    def load(self, model_dir=None):
        if not os.path.exists(model_dir):
            raise Exception("Directory to load pytorch model doesn't exist")
        self.generator = torch.load("{0}/generator.pth".format(model_dir))
        self.discriminator = torch.load("{0}/discriminator.pth".format(model_dir))
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)

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
        self.discriminator.eval()
        self.generator.eval()
        noise = gen_noise((batch_size, self.noise_dim)).to(self.device)

        attributes, data_feature = next(iter(self.real_train_dl))
        attributes = attributes.to(self.device)
        attributes = attributes[:batch_size, :]
        input_gen = torch.cat((attributes, noise), dim=1)
        with torch.no_grad():
            features = self.generator(input_gen)
            features = torch.reshape(features, (batch_size,
                                                self.real_train_dl.dataset.data_feature_shape[1],
                                                self.real_train_dl.dataset.data_feature_shape[2]))
            features = features.cpu().numpy()
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

    def train(self, epochs, writer_frequency=1, saver_frequency=20):
        avg_mmd = []
        for epoch in range(epochs):
            mmd = []
            for batch_idx, (data_attribute, data_feature) in enumerate(self.real_train_dl):
                data_attribute = data_attribute.to(self.device)
                data_feature = data_feature.to(self.device)
                data_feature = torch.flatten(data_feature, start_dim=1, end_dim=2)
                batch_size = data_attribute.shape[0]
                real = torch.cat((data_attribute, data_feature), dim=1)
                noise = gen_noise((batch_size, self.noise_dim)).to(self.device)
                input_gen = torch.cat((data_attribute, noise), dim=1)

                ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
                fake = self.generator(input_gen)
                mmd_fake = torch.reshape(fake, (batch_size,
                                                self.real_train_dl.dataset.data_feature_shape[1],
                                                self.real_train_dl.dataset.data_feature_shape[2]))
                mmd_real = torch.reshape(data_feature, (batch_size,
                                                        self.real_train_dl.dataset.data_feature_shape[1],
                                                        self.real_train_dl.dataset.data_feature_shape[2]))
                mmd.append(calculate_mmd_rbf(torch.mean(mmd_fake, dim=0).detach().cpu().numpy(),
                                             torch.mean(mmd_real, dim=0).detach().cpu().numpy()))
                fake = torch.cat((data_attribute, fake), dim=1)
                disc_real = self.discriminator(real).view(-1)
                lossD_real = self.criterion(disc_real, torch.ones_like(disc_real))
                disc_fake = self.discriminator(fake).view(-1)
                lossD_fake = self.criterion(disc_fake, torch.zeros_like(disc_fake))
                lossD = (lossD_real + lossD_fake) / 2
                self.discriminator.zero_grad()
                lossD.backward(retain_graph=True)
                self.optimizer_dis.step()

                ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
                output = self.discriminator(fake).view(-1)
                lossG = self.criterion(output, torch.ones_like(output))
                self.generator.zero_grad()
                lossG.backward()
                self.optimizer_gen.step()
            self.time_logger.info('END OF EPOCH {0}'.format(epoch))
            if epoch % saver_frequency == 0:
                self.save(epoch)
                self.inference(epoch)
        np.save("{}/mmd.npy".format(self.checkpoint_dir), np.asarray(avg_mmd))
