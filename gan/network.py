import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from output import OutputType, Normalization
from sklearn import metrics


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_feature, input_attribute, num_layers=5, num_units=200,
                 scope_name="discriminator", *args, **kwargs):
        super(Discriminator, self).__init__()
        # only saved for adding to summary writer (see trainer.train)
        self.input_size = input_feature.shape[1] * input_feature.shape[2] + input_attribute.shape[1]
        modules = []
        modules.append(nn.Linear(self.input_size, num_units))
        modules.append(nn.ReLU())
        for i in range(num_layers - 2):
            modules.append(nn.Linear(num_units, num_units))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(num_units, 1))
        # https://discuss.pytorch.org/t/append-for-nn-sequential-or-directly-converting-nn-modulelist-to-nn-sequential/7104
        self.disc = nn.Sequential(*modules)

    def forward(self, x):
        return self.disc(x)


class AttrDiscriminator(nn.Module):
    def __init__(self, input_attribute, num_layers=5, num_units=200, scope_name="attrDiscriminator", *args, **kwargs):
        super(AttrDiscriminator, self).__init__()
        # only saved for adding to summary writer (see trainer.train)
        self.input_size = input_attribute.shape[1]
        modules = []
        modules.append(nn.Linear(self.input_size, num_units))
        modules.append(nn.ReLU())
        for i in range(num_layers - 2):
            modules.append(nn.Linear(num_units, num_units))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(num_units, 1))

        self.attrdisc = nn.Sequential(*modules)

    def forward(self, x):
        return self.attrdisc(x)


class DoppelGANgerGenerator(nn.Module):
    def __init__(self, noise_dim, feature_outputs, attribute_outputs, real_attribute_mask, sample_len,
                 attribute_num_units=100, attribute_num_layers=3, feature_num_units=100,
                 feature_num_layers=1, scope_name="DoppelGANgerGenerator", *args, **kwargs):
        super(DoppelGANgerGenerator, self).__init__()

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
        modules = []
        modules.append(nn.Linear(noise_dim, attribute_num_units))
        modules.append(nn.ReLU())
        modules.append(nn.BatchNorm1d(attribute_num_units))
        for i in range(attribute_num_layers - 2):
            modules.append(nn.Linear(attribute_num_units, attribute_num_units))
            modules.append(nn.ReLU())
            modules.append(nn.BatchNorm1d(attribute_num_units))



        self.real_attribute_gen = nn.Sequential(*modules)
        self.real_attr_output_layers = []
        self.addi_attr_output_layers = []
        for i in range(len(attribute_outputs)):
            modules = []
            modules.append(nn.Linear(attribute_num_units, attribute_outputs[i].dim))
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

        modules = []
        modules.append(nn.Linear(noise_dim + self.real_attribute_dim, attribute_num_units))
        modules.append(nn.ReLU())
        modules.append(nn.BatchNorm1d(attribute_num_units))
        for i in range(attribute_num_layers - 2):
            modules.append(nn.Linear(attribute_num_units, attribute_num_units))
            modules.append(nn.ReLU())
            modules.append(nn.BatchNorm1d(attribute_num_units))

        self.addi_attribute_gen = nn.Sequential(*modules)

        # feature generator
        self.feature_rnn = nn.LSTM(input_size=noise_dim + self.real_attribute_dim + self.addi_attribute_dim,
                                   hidden_size=feature_num_units,
                                   num_layers=feature_num_layers,
                                   batch_first=True)

        self.feature_output_layers = []
        feature_counter = 0
        feature_len = len(feature_outputs)
        for i in range(len(feature_outputs) * sample_len):
            modules = []
            modules.append(nn.Linear(feature_num_units, feature_outputs[feature_counter].dim))
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

    def forward(self, real_attribute_noise, addi_attribute_noise, feature_input_noise):
        # real attribute generator
        real_attribute_gen_output = self.real_attribute_gen(real_attribute_noise)
        real_attribute_output = torch.zeros((real_attribute_noise.size(0), 0))
        real_attribute_output_discrete = torch.zeros((real_attribute_noise.size(0), 0))
        for attr_layer in self.real_attr_output_layers:
            output = attr_layer(real_attribute_gen_output)

            if isinstance(attr_layer[-1], nn.Softmax):
                one_hot = F.one_hot(torch.argmax(output, dim=1), num_classes=output.shape[1])
                real_attribute_output_discrete = torch.cat((real_attribute_output_discrete, one_hot), dim=1)
            else:
                real_attribute_output_discrete = torch.cat((real_attribute_output_discrete, output), dim=1)

            real_attribute_output = torch.cat((real_attribute_output, output), dim=1)


        # create addi attribute generator input
        addi_attribute_input = torch.cat((real_attribute_output, addi_attribute_noise), dim=1)

        # add attribute generator
        addi_attribute_gen_output = self.addi_attribute_gen(addi_attribute_input)
        addi_attribute_output = torch.zeros((real_attribute_noise.size(0), 0))
        addi_attribute_output_discrete = torch.zeros((real_attribute_noise.size(0), 0))
        for addi_attr_layer in self.addi_attr_output_layers:
            output = addi_attr_layer(addi_attribute_gen_output)

            if isinstance(addi_attr_layer[-1], nn.Softmax):
                one_hot = F.one_hot(torch.argmax(output, dim=1), num_classes=output.shape[1])
                addi_attribute_output_discrete = torch.cat((addi_attribute_output_discrete, one_hot), dim=1)
            else:
                addi_attribute_output_discrete = torch.cat((addi_attribute_output_discrete, output), dim=1)

            addi_attribute_output = torch.cat((addi_attribute_output, output), dim=1)

        # create feature generator input
        attribute_output = torch.unsqueeze(
            torch.cat((real_attribute_output_discrete, addi_attribute_output_discrete), dim=1), dim=1)
        #attribute_output = torch.unsqueeze(
         #   torch.cat((real_attribute_output, addi_attribute_output), dim=1), dim=1)
        attribute_feature_input = torch.cat(feature_input_noise.shape[1] * [attribute_output], dim=1)
        attribute_feature_input = attribute_feature_input.detach()
        feature_gen_input = torch.cat((attribute_feature_input, feature_input_noise), dim=2)

        # initial hidden and cell state
        h_o = torch.randn((self.feature_num_layers, feature_gen_input.size(0), self.feature_num_units))
        c_0 = torch.randn((self.feature_num_layers, feature_gen_input.size(0), self.feature_num_units))
        # feature generator
        feature_rnn_output, _ = self.feature_rnn(feature_gen_input, (h_o, c_0))

        feature_gen_output = torch.zeros((feature_rnn_output.size(0), feature_rnn_output.size(1), 0))
        for feature_output_layer in self.feature_output_layers:
            output = feature_output_layer(feature_rnn_output)
            feature_gen_output = torch.cat((feature_gen_output, output), dim=2)

        feature_gen_output = torch.reshape(feature_gen_output, (attribute_output.shape[0],
                                                                int((feature_gen_output.shape[1] *
                                                                     feature_gen_output.shape[
                                                                         2]) / self.feature_dim), self.feature_dim))

        return real_attribute_output, addi_attribute_output, feature_gen_output
