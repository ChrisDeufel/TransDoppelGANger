import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from output import OutputType, Normalization
from sklearn import metrics
from gan.gan_util import init_weights
import logging

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_feature, input_attribute, num_layers=5, num_units=200,
                 scope_name="discriminator", *args, **kwargs):
        super(Discriminator, self).__init__()
        # only saved for adding to summary writer (see trainer.train)
        self.input_feature_shape = input_feature.shape
        self.input_attribute_shape = input_attribute.shape
        self.input_size = input_feature.shape[1] * input_feature.shape[2] + input_attribute.shape[1]
        modules = [nn.Linear(self.input_size, num_units), nn.ReLU()]
        for i in range(num_layers - 2):
            modules.append(nn.Linear(num_units, num_units))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(num_units, 1))
        # https://discuss.pytorch.org/t/append-for-nn-sequential-or-directly-converting-nn-modulelist-to-nn-sequential/7104
        self.disc = nn.Sequential(*modules)
        # initialize weights
        self.disc.apply(init_weights)

    def forward(self, input_feature, input_attribute):
        input_feature = torch.flatten(input_feature, start_dim=1, end_dim=2)
        input_attribute = torch.flatten(input_attribute, start_dim=1, end_dim=1)
        x = torch.cat((input_feature, input_attribute), dim=1)
        return self.disc(x)


class AttrDiscriminator(nn.Module):
    def __init__(self, input_attribute, num_layers=5, num_units=200, scope_name="attrDiscriminator", *args, **kwargs):
        super(AttrDiscriminator, self).__init__()
        # only saved for adding to summary writer (see trainer.train)
        self.input_size = input_attribute.shape[1]
        modules = [nn.Linear(self.input_size, num_units), nn.ReLU()]
        for i in range(num_layers - 2):
            modules.append(nn.Linear(num_units, num_units))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(num_units, 1))

        self.attrdisc = nn.Sequential(*modules)
        # initialize weights
        self.attrdisc.apply(init_weights)

    def forward(self, x):
        return self.attrdisc(x)


class DoppelGANgerGenerator(nn.Module):
    def __init__(self, noise_dim, feature_outputs, attribute_outputs, real_attribute_mask, sample_len,
                 attribute_num_units=100, attribute_num_layers=3, feature_num_units=100,
                 feature_num_layers=1, scope_name="DoppelGANgerGenerator", *args, **kwargs):
        super(DoppelGANgerGenerator, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.feature_num_units = feature_num_units
        self.feature_num_layers = feature_num_layers
        # calculate dimensions
        self.feature_dim = 0
        self.real_attribute_dim = 0
        self.addi_attribute_dim = 0
        for feature in feature_outputs:
            self.feature_dim += feature.dim
        for i in range(len(attribute_outputs)):
            if real_attribute_mask[i]:
                self.real_attribute_dim += attribute_outputs[i].dim
            else:
                self.addi_attribute_dim += attribute_outputs[i].dim

        # build real attribute generator
        modules = [nn.Linear(noise_dim, attribute_num_units), nn.ReLU(),
                   nn.BatchNorm1d(num_features=attribute_num_units, eps=1e-5, momentum=0.9)]
        for i in range(attribute_num_layers - 2):
            modules.append(nn.Linear(attribute_num_units, attribute_num_units))
            modules.append(nn.ReLU())
            modules.append(nn.BatchNorm1d(num_features=attribute_num_units, eps=1e-5, momentum=0.9))
        self.real_attribute_gen = nn.Sequential(*modules)
        # initialize weights
        self.real_attribute_gen.apply(init_weights)

        # build additive attribute generator
        modules = [nn.Linear(noise_dim + self.real_attribute_dim, attribute_num_units), nn.ReLU(),
                   nn.BatchNorm1d(num_features=attribute_num_units, eps=1e-5, momentum=0.9)]
        for i in range(attribute_num_layers - 2):
            modules.append(nn.Linear(attribute_num_units, attribute_num_units))
            modules.append(nn.ReLU())
            modules.append(nn.BatchNorm1d(num_features=attribute_num_units, eps=1e-5, momentum=0.9))
        self.addi_attribute_gen = nn.Sequential(*modules)
        # initialize weights
        self.addi_attribute_gen.apply(init_weights)

        # create real and additive generator output layers
        self.real_attr_output_layers = nn.ModuleList()
        self.addi_attr_output_layers = nn.ModuleList()
        for i in range(len(attribute_outputs)):
            modules = [nn.Linear(attribute_num_units, attribute_outputs[i].dim)]
            if attribute_outputs[i].type_ == OutputType.DISCRETE:
                modules.append(nn.Softmax(dim=-1))
            else:
                if attribute_outputs[i].normalization == Normalization.ZERO_ONE:
                    modules.append(nn.Sigmoid())
                else:
                    modules.append(nn.Tanh())
            if real_attribute_mask[i]:
                self.real_attr_output_layers.append(nn.Sequential(*modules))
            else:
                self.addi_attr_output_layers.append(nn.Sequential(*modules))
        # initialize weights
        self.real_attr_output_layers.apply(init_weights)
        self.addi_attr_output_layers.apply(init_weights)

        # create feature generator
        self.feature_rnn = nn.LSTM(input_size=noise_dim + self.real_attribute_dim + self.addi_attribute_dim,
                                   hidden_size=feature_num_units,
                                   num_layers=feature_num_layers,
                                   batch_first=True)
        # initialize weights
        self.feature_rnn.apply(init_weights)
        # create feature output layers
        self.feature_output_layers = nn.ModuleList()
        feature_counter = 0
        feature_len = len(feature_outputs)
        for i in range(len(feature_outputs) * sample_len):
            modules = [nn.Linear(feature_num_units, feature_outputs[feature_counter].dim)]
            if feature_outputs[feature_counter].type_ == OutputType.DISCRETE:
                modules.append(nn.Softmax(dim=-1))
            else:
                if feature_outputs[feature_counter].normalization == Normalization.ZERO_ONE:
                    modules.append(nn.Sigmoid())
                else:
                    modules.append(nn.Tanh())
            feature_counter += 1
            if feature_counter % feature_len == 0:
                feature_counter = 0
            self.feature_output_layers.append(nn.Sequential(*modules))
        # initialize weights
        self.feature_output_layers.apply(init_weights)

    def forward(self, real_attribute_noise, addi_attribute_noise, feature_input_noise):
        all_attribute = []
        all_discrete_attribute = []
        # real attribute generator
        real_attribute_gen_output = self.real_attribute_gen(real_attribute_noise)
        part_attribute = []
        part_discrete_attribute = []
        for attr_layer in self.real_attr_output_layers:
            sub_output = attr_layer(real_attribute_gen_output)
            if isinstance(attr_layer[-1], nn.Softmax):
                sub_output_discrete = F.one_hot(torch.argmax(sub_output, dim=1), num_classes=sub_output.shape[1])
            else:
                sub_output_discrete = sub_output
            part_attribute.append(sub_output)
            part_discrete_attribute.append(sub_output_discrete)
        part_attribute = torch.cat(part_attribute, dim=1)
        part_discrete_attribute = torch.cat(part_discrete_attribute, dim=1)
        part_discrete_attribute = part_discrete_attribute.detach()
        all_attribute.append(part_attribute)
        all_discrete_attribute.append(part_discrete_attribute)

        # create addi attribute generator input
        addi_attribute_input = torch.cat((part_discrete_attribute, addi_attribute_noise), dim=1)
        # addi_attribute_input = torch.cat((part_attribute, addi_attribute_noise), dim=1)
        # add attribute generator
        addi_attribute_gen_output = self.addi_attribute_gen(addi_attribute_input)
        part_attribute = []
        part_discrete_attribute = []
        for addi_attr_layer in self.addi_attr_output_layers:
            sub_output = addi_attr_layer(addi_attribute_gen_output)
            if isinstance(addi_attr_layer[-1], nn.Softmax):
                sub_output_discrete = F.one_hot(torch.argmax(sub_output, dim=1), num_classes=sub_output.shape[1])
            else:
                sub_output_discrete = sub_output
            part_attribute.append(sub_output)
            part_discrete_attribute.append(sub_output_discrete)
        part_attribute = torch.cat(part_attribute, dim=1)
        part_discrete_attribute = torch.cat(part_discrete_attribute, dim=1)
        part_discrete_attribute = part_discrete_attribute.detach()
        all_attribute.append(part_attribute)
        all_discrete_attribute.append(part_discrete_attribute)
        all_attribute = torch.cat(all_attribute, dim=1)
        all_discrete_attribute = torch.cat(all_discrete_attribute, dim=1)

        # create feature generator input
        attribute_output = torch.unsqueeze(all_discrete_attribute, dim=1)
        # attribute_output = torch.unsqueeze(all_attribute, dim=1)
        attribute_feature_input = torch.cat(feature_input_noise.shape[1] * [attribute_output], dim=1)
        attribute_feature_input = attribute_feature_input.detach()
        feature_gen_input = torch.cat((attribute_feature_input, feature_input_noise), dim=2)

        # initial hidden and cell state
        h_o = torch.randn((self.feature_num_layers, feature_gen_input.size(0), self.feature_num_units)).to(self.device)
        c_0 = torch.randn((self.feature_num_layers, feature_gen_input.size(0), self.feature_num_units)).to(self.device)
        # feature generator
        feature_rnn_output, _ = self.feature_rnn(feature_gen_input, (h_o, c_0))
        # feature_rnn_output, _ = self.feature_rnn(feature_gen_input)
        features = torch.zeros((feature_rnn_output.size(0), feature_rnn_output.size(1), 0)).to(self.device)
        for feature_output_layer in self.feature_output_layers:
            sub_output = feature_output_layer(feature_rnn_output)
            features = torch.cat((features, sub_output), dim=2)

        features = torch.reshape(features, (attribute_output.shape[0],
                                            int((features.shape[1] *
                                                 features.shape[
                                                     2]) / self.feature_dim), self.feature_dim))

        return all_attribute, features
