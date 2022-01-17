import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from gan.rcgan import RGANGenerator, RGANDiscriminator
from gan.gan_util import gen_noise
import numpy as np
import torch.nn.functional as F
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


class RCGAN:
    """RCGAN Class
    """

    @property
    def name(self):
        return 'RCGAN'

    def __init__(self,
                 train_loader,
                 device,
                 lr=0.1,
                 noise_dim=5,
                 batch_size=28,
                 hidden_size_gen=100,
                 num_layer_gen=1,
                 hidden_size_dis=100,
                 num_layer_dis=1,
                 beta1=0.5,
                 checkpoint_dir="",
                 time_logging_file="",
                 config_logging_file="",
                 isConditional=True):
        self.config_logger, self.time_logger = setup_logging(time_logging_file, config_logging_file)
        # Initalize variables.
        self.real_train_dl = train_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.isConditional = isConditional

        # initiate models
        # calculate variables
        # TODO: what do we need sequence length for?
        self.sequence_length = train_loader.dataset.data_feature.shape[1]
        self.noise_dim = noise_dim
        self.attribute_size = train_loader.dataset.data_attribute.shape[1]
        num_features = train_loader.dataset.data_feature.shape[2]
        if isConditional:
            self.generator = RGANGenerator(sequence_length=self.sequence_length,
                                           output_size=num_features,
                                           hidden_size=hidden_size_gen, noise_size=self.noise_dim + self.attribute_size,
                                           num_layers=num_layer_gen)
            self.discriminator = RGANDiscriminator(sequence_length=self.sequence_length,
                                                   input_size=self.attribute_size + num_features,
                                                   hidden_size=hidden_size_dis, num_layers=num_layer_dis)
        else:
            self.generator = RGANGenerator(sequence_length=self.sequence_length, output_size=num_features,
                                           hidden_size=hidden_size_gen, noise_size=self.noise_dim,
                                           num_layers=num_layer_gen)
            self.discriminator = RGANDiscriminator(sequence_length=self.sequence_length, input_size=num_features,
                                                   hidden_size=hidden_size_dis, num_layers=num_layer_dis)
        self.config_logger.info("DISCRIMINATOR: {0}".format(self.discriminator))
        self.generator = self.generator.to(self.device)
        self.config_logger.info("GENERATOR: {0}".format(self.generator))
        self.discriminator = self.discriminator.to(self.device)
        # loss
        self.criterion = nn.BCELoss()
        self.config_logger.info("Criterion: {0}".format(self.criterion))
        # Setup optimizer
        self.optimizer_dis = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.config_logger.info("DISCRIMINATOR OPTIMIZER: {0}".format(self.optimizer_dis))
        self.optimizer_gen = optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.config_logger.info("GENERATOR OPTIMIZER: {0}".format(self.optimizer_gen))
        self.batch_size = batch_size
        self.config_logger.info("Batch Size: {0}".format(self.batch_size))
        self.config_logger.info("Noise Dimension: {0}".format(self.noise_dim))
        self.config_logger.info("d_rounds: {0}".format("1"))
        self.config_logger.info("g_rounds: {0}".format("1"))
        self.config_logger.info("Is Conditional: {0}".format(isConditional))
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
                                     self.real_train_dl.dataset.data_feature_shape[2]-2))
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
        noise = gen_noise((batch_size, self.sequence_length, self.noise_dim)).to(self.device)
        if self.isConditional:
            attributes, data_feature = next(iter(self.real_train_dl))
            attributes = attributes.to(self.device)
            attributes = attributes[:batch_size, :]
            data_attribute_noise = torch.unsqueeze(attributes, dim=1)
            data_attribute_noise = torch.cat(noise.shape[1] * [data_attribute_noise], dim=1)
            noise = torch.cat((data_attribute_noise, noise), dim=2)
        else:
            attributes = torch.zeros((batch_size, self.real_train_dl.dataset.data_attribute.shape[1]))
        with torch.no_grad():
            features = self.generator(noise)
            features = features.cpu().numpy()
            gen_flags = np.zeros(features.shape[:-1])
            lengths = np.zeros(features.shape[0])
            for i in range(len(features)):
                winner = (features[i, :, -1] > features[i, :, -2])
                argmax = np.argmax(winner==True)
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
                input_feature = data_feature.to(self.device)
                batch_size = data_attribute.shape[0]
                ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
                noise = gen_noise((batch_size, self.sequence_length, self.noise_dim)).to(self.device)
                if self.isConditional:
                    data_attribute = torch.unsqueeze(data_attribute, dim=1)
                    data_attribute = torch.cat(noise.shape[1] * [data_attribute], dim=1)
                    noise = torch.cat((data_attribute, noise), dim=2)
                    input_feature = torch.cat((data_attribute, input_feature), dim=2)
                fake = self.generator(noise)
                mmd.append(calculate_mmd_rbf(torch.mean(fake, dim=0).detach().cpu().numpy(),
                                             torch.mean(data_feature, dim=0).detach().cpu().numpy()))
                if self.isConditional:
                    fake = torch.cat((data_attribute, fake), dim=2)
                disc_real = self.discriminator(input_feature).view(-1)
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