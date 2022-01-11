import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from gan.cgan import CGANGenerator, CGANDiscriminator
from gan.gan_util import gen_noise
import numpy as np

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


class CGAN:

    @property
    def name(self):
        return 'CGAN'

    def __init__(self,
                 train_loader,
                 device,
                 lr=0.001,
                 noise_dim=30,
                 num_units=200,
                 num_layers=3,
                 beta1=0.5,
                 alpha=0.1,
                 checkpoint_dir="",
                 isWasserstein=True):
        # setup models
        self.device = device
        self.train_dl = train_loader
        self.checkpoint_dir = checkpoint_dir
        self.isWasserstein = isWasserstein
        self.noise_dim = noise_dim
        self.generator = CGANGenerator(input_feature_shape=train_loader.dataset.data_feature_shape,
                                       input_attribute_shape=train_loader.dataset.data_attribute_shape,
                                       noise_dim=noise_dim, num_units=num_units, num_layers=num_layers, alpha=alpha)
        self.discriminator = CGANDiscriminator(input_feature_shape=train_loader.dataset.data_feature_shape,
                                               input_attribute_shape=train_loader.dataset.data_attribute_shape,
                                               num_units=num_units, num_layers=num_layers, alpha=alpha)
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
        noise = gen_noise((batch_size, self.noise_dim)).to(self.device)

        attributes, data_feature = next(iter(self.train_dl))
        attributes = attributes.to(self.device)
        attributes = attributes[:batch_size, :]
        input_gen = torch.cat((attributes, noise), dim=1)
        with torch.no_grad():
            features = self.generator(input_gen)
            features = torch.reshape(features, (batch_size,
                                                self.train_dl.dataset.data_feature_shape[1],
                                                self.train_dl.dataset.data_feature_shape[2]))
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
        for epoch in range(epochs):
            for batch_idx, (data_attribute, data_feature) in enumerate(self.train_dl):
                data_attribute = data_attribute.to(self.device)
                data_feature = data_feature.to(self.device)
                data_feature = torch.flatten(data_feature, start_dim=1, end_dim=2)
                batch_size = data_attribute.shape[0]
                real = torch.cat((data_attribute, data_feature), dim=1)
                noise = gen_noise((batch_size, self.noise_dim)).to(self.device)
                input_gen = torch.cat((data_attribute, noise), dim=1)
                ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
                fake = self.generator(input_gen)
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
            time_logger.info('END OF EPOCH {0}'.format(epoch))
            if epoch % saver_frequency == 0:
                self.save(epoch)
