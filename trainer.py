import numpy as np
from statistics import mean
import torch.nn as nn
import torch.optim as optim
import torch
# from torch.utils.tensorboard import SummaryWriter
import os
from gan.network import AttrDiscriminator, Discriminator, DoppelGANgerGeneratorRNN, DoppelGANgerGeneratorAttention, \
    TransformerDiscriminator
from gan.cgan import CGANGenerator, CGANDiscriminator
from gan.rcgan import RGANGenerator, RGANDiscriminator, RCGANGenerator2, RCGANDiscriminator2
from gan.timegan import TGEncoder, TGRecovery, TGGenerator, TGDiscriminator, TGSupervisor
from gan.naivegan import NaiveGanGenerator, NaiveGanDiscriminator

from gan.gan_util import gen_noise
from util import calculate_mmd_rbf, setup_logging


class DoppelGANger:
    def __init__(self,
                 real_train_dl,
                 device,
                 checkpoint_dir='',
                 time_logging_file='',
                 config_logging_file='',
                 noise_dim=5,
                 sample_len=10,
                 batch_size=100,
                 dis_lambda_gp=10,
                 attr_dis_lambda_gp=10,
                 g_attr_d_coe=1,
                 d_rounds=1,
                 g_rounds=1,
                 gen_type='RNN',
                 dis_type='normal',
                 att_dim=50,
                 num_heads=5,
                 g_lr=0.00001,
                 g_beta1=0.5,
                 d_lr=0.00001,
                 d_beta1=0.5,
                 attr_d_lr=0.00001,
                 attr_d_beta1=0.5
                 ):
        self.config_logger, self.time_logger = setup_logging(time_logging_file, config_logging_file)
        # setup models
        if dis_type == 'MLP':
            self.dis = Discriminator(real_train_dl.dataset.data_feature_shape,
                                     real_train_dl.dataset.data_attribute_shape)
        else:
            self.dis = TransformerDiscriminator(input_feature_shape=real_train_dl.dataset.data_feature_shape,
                                                input_attribute_shape=real_train_dl.dataset.data_attribute_shape)
        self.config_logger.info("DISCRIMINATOR: {0}".format(self.dis))
        self.attr_dis = AttrDiscriminator(real_train_dl.dataset.data_attribute_shape)
        self.config_logger.info("ATTRIBUTE DISCRIMINATOR: {0}".format(self.attr_dis))

        if gen_type == 'RNN':
            noise_dim = noise_dim
        else:
            noise_dim = att_dim - real_train_dl.dataset.data_attribute.shape[1]

        if gen_type == "RNN":
            self.gen = DoppelGANgerGeneratorRNN(noise_dim=noise_dim,
                                                feature_outputs=real_train_dl.dataset.data_feature_outputs,
                                                attribute_outputs=real_train_dl.dataset.data_attribute_outputs,
                                                real_attribute_mask=real_train_dl.dataset.real_attribute_mask,
                                                device=device,
                                                sample_len=sample_len)
        else:
            self.gen = DoppelGANgerGeneratorAttention(noise_dim=noise_dim,
                                                      feature_outputs=real_train_dl.dataset.data_feature_outputs,
                                                      attribute_outputs=real_train_dl.dataset.data_attribute_outputs,
                                                      real_attribute_mask=real_train_dl.dataset.real_attribute_mask,
                                                      device=device,
                                                      sample_len=sample_len, num_heads=num_heads, attn_dim=att_dim)
        self.config_logger.info("GENERATOR: {0}".format(self.gen))
        self.criterion = "Wasserstein GAN with Gradient Penalty"
        self.config_logger.info("Criterion: {0}".format(self.criterion))
        # setup optimizer
        self.dis_opt = torch.optim.Adam(self.dis.parameters(), lr=d_lr, betas=(d_beta1, 0.999))
        self.config_logger.info("DISCRIMINATOR OPTIMIZER: {0}".format(self.dis_opt))
        self.attr_dis_opt = torch.optim.Adam(self.attr_dis.parameters(), lr=attr_d_lr, betas=(attr_d_beta1, 0.999))
        self.config_logger.info("ATTRIBUTE DISCRIMINATOR OPTIMIZER: {0}".format(self.attr_dis_opt))
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=g_lr, betas=(g_beta1, 0.999))
        self.config_logger.info("GENERATOR OPTIMIZER: {0}".format(self.gen_opt))
        self.real_train_dl = real_train_dl
        self.data_feature_shape = self.real_train_dl.dataset.data_feature_shape
        if self.data_feature_shape[1] % sample_len != 0:
            raise Exception("length must be a multiple of sample_len")
        self.sample_time = int(self.data_feature_shape[1] / sample_len)
        self.batch_size = batch_size
        self.config_logger.info("Batch Size: {0}".format(self.batch_size))
        self.noise_dim = noise_dim
        self.config_logger.info("Noise Dimension: {0}".format(self.noise_dim))
        self.sample_len = sample_len
        self.config_logger.info("Sample_Length: {0}".format(self.sample_len))
        self.dis_lambda_gp = dis_lambda_gp
        self.attr_dis_lambda_gp = attr_dis_lambda_gp
        self.g_attr_d_coe = g_attr_d_coe
        self.d_rounds = d_rounds
        self.g_rounds = g_rounds
        self.config_logger.info("d_rounds: {0}".format(self.d_rounds))
        self.config_logger.info("g_rounds: {0}".format(self.g_rounds))
        self.config_logger.info("d_lambda_gp_coefficient: {0}".format(self.dis_lambda_gp))
        self.config_logger.info("attr_d_lambda_gp_coefficient: {0}".format(self.attr_dis_lambda_gp))
        self.config_logger.info("g_attr_d_coe: {0}".format(self.g_attr_d_coe))

        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.writer = None
        self.device = device
        self.config_logger.info("Device: {0}".format(self.device))
        self.dis = self.dis.to(self.device)
        self.attr_dis = self.attr_dis.to(self.device)
        self.gen = self.gen.to(self.device)
        self.EPS = 1e-8

    # TODO: use helper function in gan_util
    def gen_attribute_input_noise(self, num_sample):
        return torch.randn(size=[num_sample, self.noise_dim])

    def gen_feature_input_noise(self, num_sample, length):
        return torch.randn(size=[num_sample, length, self.noise_dim])

    def save(self, epoch):
        if not os.path.exists("{0}/epoch_{1}".format(self.checkpoint_dir, epoch)):
            os.makedirs("{0}/epoch_{1}".format(self.checkpoint_dir, epoch))
        torch.save(self.dis, "{0}/epoch_{1}/discriminator.pth".format(self.checkpoint_dir, epoch))
        torch.save(self.attr_dis, "{0}/epoch_{1}/attr_discriminator.pth".format(self.checkpoint_dir, epoch))
        torch.save(self.gen, "{0}/epoch_{1}/generator.pth".format(self.checkpoint_dir, epoch))

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

    def load(self, model_dir=None):
        if not os.path.exists(model_dir):
            raise Exception("Directory to load pytorch model doesn't exist")
        self.dis = torch.load("{0}/discriminator.pth".format(model_dir))
        self.attr_dis = torch.load("{0}/attr_discriminator.pth".format(model_dir))
        self.gen = torch.load("{0}/generator.pth".format(model_dir))
        self.dis = self.dis.to(self.device)
        self.attr_dis = self.attr_dis.to(self.device)
        self.gen = self.gen.to(self.device)
        self.dis.device = self.device
        self.attr_dis.device = self.device
        self.gen.device = self.device

    def sample_from(self, batch_size, return_gen_flag_feature=False):
        real_attribute_noise = self.gen_attribute_input_noise(batch_size).to(self.device)
        addi_attribute_noise = self.gen_attribute_input_noise(batch_size).to(self.device)
        feature_input_noise = self.gen_feature_input_noise(batch_size, self.sample_time).to(self.device)
        self.dis.eval()
        self.attr_dis.eval()
        self.gen.eval()
        with torch.no_grad():
            attributes, features = self.gen(real_attribute_noise,
                                            addi_attribute_noise,
                                            feature_input_noise)
            attributes = attributes.cpu().numpy()
            features = features.cpu().numpy()

            # TODO: possible without loop?!
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
        return features, attributes, gen_flags, lengths

    def calculate_gp_dis(self, batch_size, fake_feature, data_feature, fake_attribute, data_attribute):
        alpha_dim2 = torch.FloatTensor(batch_size, 1).uniform_(1).to(self.device)
        alpha_dim3 = torch.unsqueeze(alpha_dim2, 2).to(self.device)
        differences_input_feature = (fake_feature -
                                     data_feature)
        interpolates_input_feature = (data_feature +
                                      alpha_dim3 * differences_input_feature)
        differences_input_attribute = (fake_attribute -
                                       data_attribute)
        interpolates_input_attribute = (data_attribute +
                                        (alpha_dim2 *
                                         differences_input_attribute))
        mixed_scores = self.dis(interpolates_input_feature,
                                interpolates_input_attribute)
        gradients = torch.autograd.grad(
            inputs=[interpolates_input_feature, interpolates_input_attribute],
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True
        )
        slopes1 = torch.sum(torch.square(gradients[0]),
                            dim=(1, 2))
        slopes2 = torch.sum(torch.square(gradients[1]),
                            dim=(1))
        slopes = torch.sqrt(slopes1 + slopes2 + self.EPS)
        loss_dis_gp = torch.mean((slopes - 1.) ** 2)
        loss_dis_gp_unflattened = (slopes - 1.) ** 2
        return loss_dis_gp, loss_dis_gp_unflattened

    def calculate_gp_attr_dis(self, batch_size, fake_attribute, data_attribute):
        alpha_dim2 = torch.FloatTensor(batch_size, 1).uniform_(1).to(self.device)
        differences_input_attribute = (fake_attribute -
                                       data_attribute)
        interpolates_input_attribute = (data_attribute +
                                        (alpha_dim2 *
                                         differences_input_attribute))
        mixed_scores = self.attr_dis(interpolates_input_attribute)
        gradients = torch.autograd.grad(
            inputs=interpolates_input_attribute,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True
        )
        slopes1 = torch.sum(torch.square(gradients[0]),
                            dim=(1))
        slopes = torch.sqrt(slopes1 + self.EPS)
        loss_attr_dis_gp = torch.mean((slopes - 1.) ** 2)
        loss_attr_dis_gp_unflattened = (slopes - 1.) ** 2
        return loss_attr_dis_gp, loss_attr_dis_gp_unflattened

    def add_losses(self, running_losses, writer_frequency, epoch, n_total_steps, batch_idx):
        self.writer.add_scalar('loss/d_wo_gp', running_losses["dis_wogp_rl"] / writer_frequency,
                               epoch * n_total_steps + batch_idx)
        self.writer.add_scalar('loss/d_total', running_losses["dis_total_rl"] / writer_frequency,
                               epoch * n_total_steps + batch_idx)
        self.writer.add_scalar('loss/d/fake', running_losses["dis_fake_rl"] / writer_frequency,
                               epoch * n_total_steps + batch_idx)
        self.writer.add_scalar('loss/d/real', running_losses["dis_real_rl"] / writer_frequency,
                               epoch * n_total_steps + batch_idx)
        self.writer.add_scalar('loss/d/gp', running_losses["dis_gp_rl"] / writer_frequency,
                               epoch * n_total_steps + batch_idx)
        self.writer.add_scalar('loss/attr_d', running_losses["attr_dis_rl"] / writer_frequency,
                               epoch * n_total_steps + batch_idx)
        self.writer.add_scalar('loss/attr_d/fake', running_losses["attr_dis_fake_rl"] / writer_frequency,
                               epoch * n_total_steps + batch_idx)
        self.writer.add_scalar('loss/attr_d/real', running_losses["attr_dis_real_rl"] / writer_frequency,
                               epoch * n_total_steps + batch_idx)
        self.writer.add_scalar('loss/attr_d/gp', running_losses["attr_dis_gp_rl"] / writer_frequency,
                               epoch * n_total_steps + batch_idx)
        self.writer.add_scalar('loss/g/d', running_losses["gen_d_rl"] / writer_frequency,
                               epoch * n_total_steps + batch_idx)
        self.writer.add_scalar('loss/g/attr_d', running_losses["gen_attr_d_rl"] / writer_frequency,
                               epoch * n_total_steps + batch_idx)
        self.writer.add_scalar('loss/g', running_losses["gen_rl"] / writer_frequency,
                               epoch * n_total_steps + batch_idx)
        running_losses["dis_total_rl"] = 0
        running_losses["dis_wogp_rl"] = 0
        running_losses["dis_fake_rl"] = 0
        running_losses["dis_real_rl"] = 0
        running_losses["dis_gp_rl"] = 0
        running_losses["attr_dis_rl"] = 0
        running_losses["attr_dis_fake_rl"] = 0
        running_losses["attr_dis_real_rl"] = 0
        running_losses["attr_dis_gp_rl"] = 0
        running_losses["gen_d_rl"] = 0
        running_losses["gen_rl"] = 0
        running_losses["gen_attr_d_rl"] = 0
        return running_losses

    def train(self, epochs, writer_frequency=1, saver_frequency=10):
        self.dis.train()
        self.attr_dis.train()
        self.gen.train()

        # add models to writer
        # self.writer.add_graph(model=self.dis, input_to_model=[
        #     torch.randn([1, self.dis.input_feature_shape[1], self.dis.input_feature_shape[2]]).to(self.device),
        #     torch.randn([1, self.dis.input_attribute_shape[1]]).to(self.device)])
        # self.writer.add_graph(self.attr_dis, input_to_model=torch.randn([1, self.attr_dis.input_size]).
        # to(self.device))
        # self.writer.add_graph(self.gen,
        #                       input_to_model=[torch.randn(1, self.noise_dim).to(self.device),
        #                                       torch.randn(1, self.noise_dim).to(self.device),
        #                                       torch.randn(1, self.sample_time, self.noise_dim).to(self.device)])

        # create all running losses (rl) dict
        # running_losses = {
        #     "dis_total_rl": 0,
        #     "dis_wogp_rl": 0,
        #     "dis_fake_rl": 0,
        #     "dis_real_rl": 0,
        #     "dis_gp_rl": 0,
        #     "attr_dis_rl": 0,
        #     "attr_dis_fake_rl": 0,
        #     "attr_dis_real_rl": 0,
        #     "attr_dis_gp_rl": 0,
        #     "gen_rl": 0,
        #     "gen_d_rl": 0,
        #     "gen_attr_d_rl": 0
        # }
        avg_mmd = []
        n_total_steps = len(self.real_train_dl)
        for epoch in range(epochs):
            self.dis.train()
            self.attr_dis.train()
            self.gen.train()
            mmd = []
            for batch_idx, (data_attribute, data_feature) in enumerate(self.real_train_dl):
                data_attribute = data_attribute.to(self.device)
                data_feature = data_feature.to(self.device)
                batch_size = data_attribute.shape[0]
                # Train Critic: max E[critic(real)] - E[critic(fake)]
                for _ in range(self.d_rounds):
                    real_attribute_noise = self.gen_attribute_input_noise(batch_size).to(self.device)
                    addi_attribute_noise = self.gen_attribute_input_noise(batch_size).to(self.device)
                    feature_input_noise = self.gen_feature_input_noise(batch_size, self.sample_time).to(self.device)
                    fake_attribute, fake_feature = self.gen(real_attribute_noise,
                                                            addi_attribute_noise,
                                                            feature_input_noise)
                    mmd.append(calculate_mmd_rbf(torch.mean(fake_feature, dim=0).detach().cpu().numpy(),
                                                 torch.mean(data_feature, dim=0).detach().cpu().numpy()))
                    # discriminator
                    dis_real = self.dis(data_feature, data_attribute)
                    dis_fake = self.dis(fake_feature, fake_attribute)

                    loss_dis_fake = torch.mean(dis_fake)
                    loss_dis_real = -torch.mean(dis_real)
                    # running_losses["dis_wogp_rl"] += (loss_dis_fake + loss_dis_real).item()

                    # calculate gradient penalty
                    loss_dis_gp, loss_dis_gp_unflattened = self.calculate_gp_dis(batch_size, fake_feature, data_feature,
                                                                                 fake_attribute, data_attribute)
                    loss_dis = loss_dis_fake + loss_dis_real + self.dis_lambda_gp * loss_dis_gp
                    self.dis.zero_grad()
                    loss_dis.backward(retain_graph=True)
                    self.dis_opt.step()

                    # running_losses["dis_total_rl"] += loss_dis.item()
                    # running_losses["dis_fake_rl"] += loss_dis_fake.item()
                    # running_losses["dis_real_rl"] += loss_dis_real.item()
                    # running_losses["dis_gp_rl"] += loss_dis_gp.item()

                    # attribute discriminator
                    attr_dis_real = self.attr_dis(data_attribute)
                    attr_dis_fake = self.attr_dis(fake_attribute)
                    loss_attr_dis_real = -torch.mean(attr_dis_real)
                    loss_attr_dis_fake = torch.mean(attr_dis_fake)
                    # calculate gradient penalty
                    loss_attr_dis_gp, loss_attr_dis_gp_unflattened = self.calculate_gp_attr_dis(batch_size,
                                                                                                fake_attribute,
                                                                                                data_attribute)
                    loss_attr_dis = loss_attr_dis_fake + loss_attr_dis_real + self.attr_dis_lambda_gp * loss_attr_dis_gp

                    self.attr_dis.zero_grad()
                    loss_attr_dis.backward(retain_graph=True)
                    self.attr_dis_opt.step()

                    # running_losses["attr_dis_rl"] += loss_attr_dis.item()
                    # running_losses["attr_dis_fake_rl"] += loss_attr_dis_fake.item()
                    # running_losses["attr_dis_real_rl"] += loss_attr_dis_real.item()
                    # running_losses["attr_dis_gp_rl"] += loss_attr_dis_gp.item()

                # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
                for _ in range(self.g_rounds):
                    gen_d_fake = self.dis(fake_feature, fake_attribute)
                    gen_attr_d_fake = self.attr_dis(fake_attribute)
                    loss_gen_d = -torch.mean(gen_d_fake)
                    loss_gen_attr_d = -torch.mean(gen_attr_d_fake)
                    loss_gen = loss_gen_d + self.g_attr_d_coe * loss_gen_attr_d
                    self.gen.zero_grad()
                    loss_gen.backward()
                    self.gen_opt.step()

                    # running_losses["gen_d_rl"] += loss_gen_d.item()
                    # running_losses["gen_attr_d_rl"] += loss_gen_attr_d.item()
                    # running_losses["gen_rl"] += loss_gen.item()
                    # write losses to summary writer
                    # if (batch_idx + 1) % writer_frequency == 0:
                    #     running_losses = self.add_losses(running_losses, writer_frequency, epoch, n_total_steps,
                    #                                      batch_idx)
            self.time_logger.info('END OF EPOCH {0}'.format(epoch))
            avg_mmd.append(np.asarray(mmd).mean())
            # save model
            if epoch % saver_frequency == 0:
                self.save(epoch)
                self.inference(epoch)
        np.save("{}/mmd.npy".format(self.checkpoint_dir), np.asarray(avg_mmd))
        # self.writer.close()


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
                 checkpoint_dir='',
                 time_logging_file='',
                 config_logging_file=''):
        self.config_logger, self.time_logger = setup_logging(time_logging_file, config_logging_file)
        # setup models
        self.device = device
        self.real_train_dl = train_loader
        self.checkpoint_dir = checkpoint_dir
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
                    gen_flags[i, :argmax + 1] = 1
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
            avg_mmd.append(mean(mmd))
            self.time_logger.info('END OF EPOCH {0}'.format(epoch))
            if epoch % saver_frequency == 0:
                self.save(epoch)
                self.inference(epoch)
        np.save("{}/mmd.npy".format(self.checkpoint_dir), np.asarray(avg_mmd))


class NAIVEGAN:

    @property
    def name(self):
        return 'NAIVEGAN'

    def __init__(self,
                 train_loader,
                 device,
                 lr=0.001,
                 noise_dim=5,
                 batch_size=20,
                 num_units=200,
                 num_layers=5,
                 beta1=0.5,
                 alpha=0.1,
                 checkpoint_dir="",
                 time_logging_file='',
                 config_logging_file=''):
        # setup models
        self.config_logger, self.time_logger = setup_logging(time_logging_file, config_logging_file)
        self.device = device
        self.real_train_dl = train_loader
        self.checkpoint_dir = checkpoint_dir
        self.noise_dim = noise_dim
        self.generator = NaiveGanGenerator(input_feature_shape=train_loader.dataset.data_feature_shape,
                                           input_attribute_shape=train_loader.dataset.data_attribute_shape,
                                           noise_dim=noise_dim, num_units=num_units, num_layers=num_layers, alpha=alpha)
        self.discriminator = NaiveGanDiscriminator(input_feature_shape=train_loader.dataset.data_feature_shape,
                                                   input_attribute_shape=train_loader.dataset.data_attribute_shape,
                                                   num_units=num_units, num_layers=num_layers, alpha=alpha)
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)
        # loss
        self.criterion = nn.BCELoss()
        # Setup optimizer
        self.optimizer_gen = optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizer_dis = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.config_logger.info("DISCRIMINATOR: {0}".format(self.discriminator))
        self.config_logger.info("GENERATOR: {0}".format(self.generator))
        self.config_logger.info("Criterion: {0}".format(self.criterion))
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

        attributes = torch.zeros((batch_size, self.real_train_dl.dataset.data_attribute.shape[1]))
        with torch.no_grad():
            features = self.generator(noise)
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
                    gen_flags[i, :argmax + 1] = 1
                lengths[i] = argmax
            if not return_gen_flag_feature:
                features = features[:, :, :-2]
        return features, attributes.cpu().numpy(), gen_flags, lengths

    def train(self, epochs, writer_frequency=1, saver_frequency=20):
        avg_mmd = []
        for epoch in range(epochs):
            mmd = []
            for batch_idx, (data_attribute, data_feature) in enumerate(self.real_train_dl):
                # data_attribute = data_attribute.to(self.device)
                data_feature = data_feature.to(self.device)
                data_feature = torch.flatten(data_feature, start_dim=1, end_dim=2)
                batch_size = data_attribute.shape[0]
                # real = torch.cat((data_attribute, data_feature), dim=1)
                noise = gen_noise((batch_size, self.noise_dim)).to(self.device)
                # input_gen = torch.cat((data_attribute, noise), dim=1)
                ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
                fake = self.generator(noise)
                mmd_fake = torch.reshape(fake, (batch_size,
                                                self.real_train_dl.dataset.data_feature_shape[1],
                                                self.real_train_dl.dataset.data_feature_shape[2]))
                mmd_real = torch.reshape(data_feature, (batch_size,
                                                        self.real_train_dl.dataset.data_feature_shape[1],
                                                        self.real_train_dl.dataset.data_feature_shape[2]))
                mmd.append(calculate_mmd_rbf(torch.mean(mmd_fake, dim=0).detach().cpu().numpy(),
                                             torch.mean(mmd_real, dim=0).detach().cpu().numpy()))
                # fake = torch.cat((data_attribute, fake), dim=1)
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
            avg_mmd.append(mean(mmd))
            self.time_logger.info('END OF EPOCH {0}'.format(epoch))
            if epoch % saver_frequency == 0:
                self.save(epoch)
                self.inference(epoch)
        np.save("{}/mmd.npy".format(self.checkpoint_dir), np.asarray(avg_mmd))


class TimeGAN:
    # TimeGAN Class

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
                 batch_size=128,
                 beta1=0.9,
                 w_lambda=1,
                 w_eta=0.1,
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
        self.w_lambda = w_lambda
        self.w_eta = w_eta
        self.checkpoint_dir = checkpoint_dir
        self.batch_size = batch_size

        # Create and initialize networks.
        # determine input size for encoder
        num_features = self.real_train_dl.dataset.data_feature.shape[2]
        self.nete = TGEncoder(input_size=num_features, hidden_dim=self.hidden_dim, num_layer=self.num_layer).to(
            self.device)
        self.netr = TGRecovery(output_size=num_features, hidden_dim=self.hidden_dim, num_layer=self.num_layer).to(
            self.device)
        self.netg = TGGenerator(z_dim=self.z_dim, hidden_dim=self.hidden_dim, num_layer=self.num_layer).to(self.device)
        self.netd = TGDiscriminator(hidden_dim=self.hidden_dim, num_layer=self.num_layer).to(
            self.device)
        self.nets = TGSupervisor(hidden_dim=self.hidden_dim, num_layer=self.num_layer).to(self.device)
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

        self.config_logger.info("EMBEDDER / RECOVERY OPTIMIZER: {0}".format(self.optimizer_er))
        self.config_logger.info("DISCRIMINATOR OPTIMIZER: {0}".format(self.optimizer_d))
        self.config_logger.info("GENERATOR / SUPERVISOR OPTIMIZER: {0}".format(self.optimizer_gs))

        self.config_logger.info("Reconstruction Loss: {0}".format(self.l_mse))
        self.config_logger.info("Unsupervised Loss: {0}".format(self.l_mse))
        self.config_logger.info("Generator Adversarial Loss: {0}".format(self.l_bce))

        self.config_logger.info("Discriminator Adversarial Loss: {0}".format(self.l_bce))

        self.config_logger.info("Batch Size: {0}".format(self.batch_size))
        self.config_logger.info("Z-Dimension: {0}".format(self.z_dim))
        self.config_logger.info("Hidden Dimension: {0}".format(self.hidden_dim))
        self.config_logger.info("Beta 1: {0}".format(self.beta1))
        self.config_logger.info("w_gamma: {0}".format(self.w_lambda))
        self.config_logger.info("w_eta: {0}".format(self.w_eta))

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
        Z = gen_noise((batch_size, self.real_train_dl.dataset.data_feature_shape[1], self.z_dim)).to(self.device)
        Z_np = Z.detach().cpu().numpy()
        E_hat = self.netg(Z)  # [?, 24, 24]
        E_hat_np = E_hat.detach().cpu().numpy()
        H_hat = self.nets(E_hat)  # [?, 24, 24]
        H_hat_np = H_hat.detach().cpu().numpy()
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
                gen_flags[i, :argmax + 1] = 1
            lengths[i] = argmax
        if not return_gen_flag_feature:
            features = features[:, :, :-2]
        return features, attributes.cpu().numpy(), gen_flags, lengths

    def generate_fake_feature(self, Z):
        E_hat = self.netg(Z)
        H_hat = self.nets(E_hat)
        return self.netr(H_hat)

    def train(self, epochs, writer_frequency=1, saver_frequency=10):
        # Train the model

        self.nete.train()
        self.netr.train()
        self.netg.train()
        self.netd.train()
        self.nets.train()
        avg_mmd = []
        for epoch in range(epochs):
            mmd = []
            for batch_idx, (data_attribute, data_feature) in enumerate(self.real_train_dl):
                # torch.autograd.set_detect_anomaly(True)
                data_feature = data_feature.to(self.device)
                # (1) Map between Feature and Latent Space
                H = self.nete(data_feature)
                H_supervise = self.nets(H)
                X_tilde = self.netr(H)
                # (2) Generate Synthetic Latent Codes
                Z = gen_noise((self.batch_size, data_feature.shape[1], self.z_dim)).to(self.device)
                # add mmd
                fake_feature = self.generate_fake_feature(Z)
                mmd.append(calculate_mmd_rbf(torch.mean(fake_feature, dim=0).detach().cpu().numpy(),
                                             torch.mean(data_feature, dim=0).detach().cpu().numpy()))
                E_hat = self.netg(Z)
                H_hat = self.nets(E_hat)
                # (3) Distinguish between Real and Synthetic Codes
                Y_real = self.netd(H)
                Y_fake = self.netd(H_hat)
                # (4) Compute Reconstruction, Unsupervised, and Supervised Losses
                # Reconstruction Loss
                err_r = self.l_mse(data_feature, X_tilde)
                # Supervised Loss
                err_s = self.l_mse(H_supervise[:, :-1, :], H[:, 1:, :])
                # Encoder and Recovery Loss
                err_er = self.w_lambda * err_s + err_r
                self.optimizer_er.zero_grad()
                # Supervisor and Generator Loss
                err_g_fake = self.l_bce(Y_fake, torch.ones_like(Y_fake))
                err_gs = self.w_eta * err_s + err_g_fake
                self.optimizer_gs.zero_grad()
                # Unsupervised Loss
                err_d_real = self.l_bce(Y_real, torch.ones_like(Y_real))
                err_d_fake = self.l_bce(Y_fake, torch.zeros_like(Y_fake))
                # Discriminator Loss
                err_d = err_d_real + err_d_fake
                self.optimizer_d.zero_grad()
                # (5) update e, r, g, d, s
                err_er.backward(retain_graph=True)
                # Generator and Supervisor
                err_gs.backward(retain_graph=True)
                # Discriminator
                err_d.backward()
                self.optimizer_er.step()
                self.optimizer_gs.step()
                self.optimizer_d.step()
            avg_mmd.append(np.asarray(mmd).mean())
            self.time_logger.info('END OF EPOCH {0}'.format(epoch))
            if epoch % saver_frequency == 0:
                self.nete.train()
                self.netr.train()
                self.netg.train()
                self.netd.train()
                self.nets.train()
                self.save(epoch)
                self.inference(epoch)
        np.save("{}/mmd.npy".format(self.checkpoint_dir), np.asarray(avg_mmd))


class RCGAN2:

    @property
    def name(self):
        return 'CGAN'

    def __init__(self,
                 train_loader,
                 device,
                 batch_size=28,
                 lr=0.0001,
                 noise_dim=30,
                 num_units_dis=100,
                 num_units_gen=200,
                 num_layers=3,
                 beta1=0.5,
                 alpha=0.1,
                 checkpoint_dir='',
                 time_logging_file='',
                 config_logging_file=''):
        self.config_logger, self.time_logger = setup_logging(time_logging_file, config_logging_file)
        # setup models
        self.device = device
        self.real_train_dl = train_loader
        self.checkpoint_dir = checkpoint_dir
        self.noise_dim = noise_dim
        self.generator = RCGANGenerator2(input_feature_shape=train_loader.dataset.data_feature_shape,
                                         input_attribute_shape=train_loader.dataset.data_attribute_shape,
                                         noise_dim=noise_dim, num_units=num_units_gen, num_layers=num_layers,
                                         alpha=alpha)
        self.discriminator = RCGANDiscriminator2(input_feature_shape=train_loader.dataset.data_feature_shape,
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
                    gen_flags[i, :argmax + 1] = 1
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
            avg_mmd.append(mean(mmd))
            self.time_logger.info('END OF EPOCH {0}'.format(epoch))
            if epoch % saver_frequency == 0:
                self.save(epoch)
                self.inference(epoch)
                self.generator.train()
                self.discriminator.train()
        np.save("{}/mmd.npy".format(self.checkpoint_dir), np.asarray(avg_mmd))


class RCGAN:
    """RCGAN Class
    """

    @property
    def name(self):
        return 'RCGAN'

    def __init__(self,
                 train_loader,
                 device,
                 lr=0.001,
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
                                           num_layers=num_layer_gen, device=self.device)
            self.discriminator = RGANDiscriminator(sequence_length=self.sequence_length,
                                                   input_size=self.attribute_size + num_features,
                                                   hidden_size=hidden_size_dis, num_layers=num_layer_dis,
                                                   device=self.device)
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
                argmax = np.argmax(winner == True)
                if argmax == 0:
                    gen_flags[i, :] = 1
                else:
                    gen_flags[i, :argmax + 1] = 1
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
            avg_mmd.append(mean(mmd))
            self.time_logger.info('END OF EPOCH {0}'.format(epoch))
            if epoch % saver_frequency == 0:
                self.save(epoch)
                self.inference(epoch)
                self.discriminator.train()
                self.generator.train()
        np.save("{}/mmd.npy".format(self.checkpoint_dir), np.asarray(avg_mmd))


class TimeGAN2:
    # TimeGAN Class

    @property
    def name(self):
        return 'TimeGAN'

    def __init__(self,
                 train_loader,
                 device,
                 lr=0.001,
                 z_dim=6,
                 hidden_dim=24,
                 batch_size=128,
                 num_layer=3,
                 beta1=0.9,
                 w_gamma=1,
                 w_es=0.1,
                 w_e0=10,
                 w_g=100,
                 g_rounds=3,
                 checkpoint_dir="",
                 config_logging_file="",
                 time_logging_file=""):
        self.config_logger, self.time_logger = setup_logging(time_logging_file, config_logging_file)
        # Initalize variables.
        self.batch_size = batch_size
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
        self.g_rounds = g_rounds
        # Create and initialize networks.
        # determine input size for encoder
        num_features = self.real_train_dl.dataset.data_feature.shape[2]
        self.nete = TGEncoder(input_size=num_features, hidden_dim=self.hidden_dim, num_layer=self.num_layer).to(
            self.device)
        self.netr = TGRecovery(output_size=num_features, hidden_dim=self.hidden_dim, num_layer=self.num_layer).to(
            self.device)
        self.netg = TGGenerator(z_dim=self.z_dim, hidden_dim=self.hidden_dim, num_layer=self.num_layer).to(self.device)
        self.netd = TGDiscriminator(hidden_dim=self.hidden_dim, num_layer=self.num_layer).to(
            self.device)
        self.nets = TGSupervisor(hidden_dim=self.hidden_dim, num_layer=self.num_layer).to(self.device)
        # loss
        self.l_mse = nn.MSELoss()
        self.l_r = nn.L1Loss()
        self.l_bce = nn.BCELoss()
        self.l_bce_with_logits = nn.BCEWithLogitsLoss()
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

        self.config_logger.info("EMBEDDER / RECOVERY OPTIMIZER: {0}".format(self.optimizer_er))
        self.config_logger.info("DISCRIMINATOR OPTIMIZER: {0}".format(self.optimizer_d))
        self.config_logger.info("GENERATOR / SUPERVISOR OPTIMIZER: {0}".format(self.optimizer_gs))

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
        self.config_logger.info("g_rounds: {0}".format(self.g_rounds))

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
        Z = gen_noise((batch_size, self.real_train_dl.dataset.data_feature_shape[1], self.z_dim)).to(self.device)
        # E_hat = self.netg(Z)
        # H_hat = self.nets(E_hat)
        # features = self.netr(H_hat)
        features = self.generate_fake_feature(Z)
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
                gen_flags[i, :argmax + 1] = 1
            lengths[i] = argmax
        if not return_gen_flag_feature:
            features = features[:, :, :-2]
        return features, attributes.cpu().numpy(), gen_flags, lengths

    def train_one_iter_er(self, data_feature):

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

        self.batch_size = data_feature.shape[0]

        # set mini-batch
        X = data_feature
        Z = gen_noise((self.batch_size, data_feature.shape[1], self.z_dim)).to(self.device)
        # with autograd.detect_anomaly():
        # autograd.set_detect_anomaly(True)
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
        # err_s.backward()
        # G loss u
        err_g_U = self.l_bce(Y_fake, torch.ones_like(Y_fake))
        # G loss v
        err_g_V1 = torch.mean(torch.abs(torch.sqrt(torch.std(X_hat, [0])[1] + 1e-6) - torch.sqrt(
            torch.std(X, [0])[1] + 1e-6)))  # |a^2 - b^2|
        err_g_V2 = torch.mean(
            torch.abs((torch.mean(X_hat, [0])[0]) - (torch.mean(X, [0])[0])))  # |a - b|
        err_g_V = err_g_V1 + err_g_V2
        # G loss ue
        err_g_U_e = self.l_bce(Y_fake_e, torch.ones_like(Y_fake_e))
        err_g = err_g_U + self.w_gamma * err_g_U_e + 100 * torch.sqrt(err_s) + 100 * err_g_V
        err_g.backward(retain_graph=True)
        self.optimizer_gs.step()

    def train_one_iter_d(self, data_feature):
        # Train the model for one epoch.
        # set mini-batch
        X = data_feature
        Z = gen_noise((self.batch_size, data_feature.shape[1], self.z_dim)).to(self.device)

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
        err_d_real = self.l_bce(Y_real, torch.ones_like(Y_real))
        err_d_fake = self.l_bce(Y_fake, torch.zeros_like(Y_fake))
        err_d_fake_e = self.l_bce(Y_fake_e, torch.ones_like(Y_fake_e))
        err_d = err_d_real + err_d_fake + err_d_fake_e * self.w_gamma
        if err_d > 0.15:
            err_d.backward(retain_graph=True)
        self.optimizer_d.step()

    def generate_fake_feature(self, Z):
        E_hat = self.netg(Z)
        H_hat = self.nets(E_hat)
        return self.netr(H_hat)

    def train(self, epochs, writer_frequency=1, saver_frequency=10):
        # Train the model
        self.nete.train()
        self.netr.train()
        self.netg.train()
        self.netd.train()
        self.nets.train()
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
        avg_mmd = []
        for iter in range(epochs * 2):
            self.nete.train()
            self.netr.train()
            self.netg.train()
            self.netd.train()
            self.nets.train()
            mmd = []
            for batch_idx, (data_attribute, data_feature) in enumerate(self.real_train_dl):
                data_feature = data_feature.to(self.device)
                Z = gen_noise((data_feature.shape[0], data_feature.shape[1], self.z_dim)).to(self.device)
                fake_feature = self.generate_fake_feature(Z)
                mmd.append(calculate_mmd_rbf(torch.mean(fake_feature, dim=0).detach().cpu().numpy(),
                                             torch.mean(data_feature, dim=0).detach().cpu().numpy()))
                for kk in range(self.g_rounds):
                    # Train Generator and Supervisor
                    self.train_one_iter_g(data_feature)
                    # Train Embedder and Recovery again
                    self.train_one_iter_er_2(data_feature)
                self.train_one_iter_d(data_feature)
            avg_mmd.append(mean(mmd))
            self.time_logger.info('Joint Training - END OF EPOCH {0}'.format(iter))
            if iter % saver_frequency == 0:
                self.nete.train()
                self.netr.train()
                self.netg.train()
                self.netd.train()
                self.nets.train()
                self.save(iter)
                self.inference(iter)
        np.save("{}/mmd.npy".format(self.checkpoint_dir), np.asarray(avg_mmd))
        self.time_logger.info('Finish Joint Training')
