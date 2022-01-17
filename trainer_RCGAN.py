import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from gan.rcgan import RGANGenerator, RGANDiscriminator
from gan.gan_util import gen_noise
import numpy as np
import torch.nn.functional as F

time_logger = logging.getLogger(__name__)
time_logger.setLevel(logging.INFO)
config_logger = logging.getLogger(__name__)
config_logger.setLevel(logging.INFO)


def add_handler_trainer(handlers):
    for handler in handlers:
        time_logger.addHandler(handler)


def add_config_handler_trainer(handlers):
    for handler in handlers:
        config_logger.addHandler(handler)


class RCGAN:
    """TimeGAN Class
    """

    @property
    def name(self):
        return 'RCGAN'

    def __init__(self,
                 train_loader,
                 device,
                 lr=0.1,
                 noise_size=5,
                 hidden_size_gen=100,
                 num_layer_gen=1,
                 hidden_size_dis=100,
                 num_layer_dis=1,
                 beta1=0.5,
                 checkpoint_dir="",
                 isConditional=True):
        # Initalize variables.
        self.train_dl = train_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.isConditional = isConditional
        config_logger.info("Learning Rate: {0}".format(lr))
        config_logger.info("Beta 1: {0}".format(beta1))

        # initiate models
        # calculate variables
        # TODO: what do we need sequence length for?
        self.sequence_length = train_loader.dataset.data_feature.shape[1]
        self.noise_size = noise_size
        self.attribute_size = train_loader.dataset.data_attribute.shape[1]
        num_features = train_loader.dataset.data_feature.shape[2]
        if isConditional:
            self.generator = RGANGenerator(sequence_length=self.sequence_length,
                                           output_size=num_features,
                                           hidden_size=hidden_size_gen, noise_size=noise_size + self.attribute_size,
                                           num_layers=num_layer_gen)
            self.discriminator = RGANDiscriminator(sequence_length=self.sequence_length,
                                                   input_size=self.attribute_size + num_features,
                                                   hidden_size=hidden_size_dis, num_layers=num_layer_dis)
        else:
            self.generator = RGANGenerator(sequence_length=self.sequence_length, output_size=num_features,
                                           hidden_size=hidden_size_gen, noise_size=noise_size, num_layers=num_layer_gen)
            self.discriminator = RGANDiscriminator(sequence_length=self.sequence_length, input_size=num_features,
                                                   hidden_size=hidden_size_dis, num_layers=num_layer_dis)
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)
        config_logger.info("GENERATOR: {0}".format(self.generator))
        config_logger.info("DISCRIMINATOR: {0}".format(self.discriminator))
        # loss
        self.criterion = nn.BCELoss()
        # Setup optimizer
        self.optimizer_gen = optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizer_dis = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

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

    def sample_from(self, batch_size, return_gen_flag_feature=False):
        self.discriminator.eval()
        self.generator.eval()
        noise = gen_noise((batch_size, self.sequence_length, self.noise_size)).to(self.device)
        if self.isConditional:
            attributes, data_feature = next(iter(self.train_dl))
            attributes = attributes.to(self.device)
            attributes = attributes[:batch_size, :]
            data_attribute_noise = torch.unsqueeze(attributes, dim=1)
            data_attribute_noise = torch.cat(noise.shape[1] * [data_attribute_noise], dim=1)
            noise = torch.cat((data_attribute_noise, noise), dim=2)
        else:
            attributes = torch.zeros((batch_size, self.train_dl.dataset.data_attribute.shape[1]))
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
            for epoch in range(epochs):
                for batch_idx, (data_attribute, data_feature) in enumerate(self.train_dl):
                    data_attribute = data_attribute.to(self.device)
                    data_feature = data_feature.to(self.device)
                    batch_size = data_attribute.shape[0]
                    ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
                    noise = gen_noise((batch_size, self.sequence_length, self.noise_size)).to(self.device)
                    if self.isConditional:
                        data_attribute = torch.unsqueeze(data_attribute, dim=1)
                        data_attribute = torch.cat(noise.shape[1] * [data_attribute], dim=1)
                        noise = torch.cat((data_attribute, noise), dim=2)
                        data_feature = torch.cat((data_attribute, data_feature), dim=2)
                    fake = self.generator(noise)
                    if self.isConditional:
                        fake = torch.cat((data_attribute, fake), dim=2)
                    disc_real = self.discriminator(data_feature).view(-1)
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
                time_logger.info('END OF EPOCH {0}'.format(epoch))
                if epoch % saver_frequency == 0:
                    self.save(epoch)
