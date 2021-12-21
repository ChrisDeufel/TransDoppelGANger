from torch.utils.data import DataLoader
import torch
import numpy as np
from trainer import Trainer
from gan.network import Discriminator, AttrDiscriminator, DoppelGANgerGeneratorRNN, DoppelGANgerGeneratorAttention
# from gan.network import Discriminator, AttrDiscriminator, DoppelGANgerGenerator
from load_data import load_data
from util import normalize_per_sample, add_gen_flag
from data import Data, LargeData, SplitData

device = "cuda" if torch.cuda.is_available() else "cpu"
dataset_name = "index_growth_12mo"
gan_type = 'RNN'

sample_len = 12
batch_size = 20
attn_dim = 100
# load data
if dataset_name == "transactions":
    dataset = SplitData(sample_len, name=dataset_name)
else:
    dataset = Data(sample_len=sample_len, name=dataset_name)
real_train_dl = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

if gan_type == 'RNN':
    noise_dim = 5
else:
    noise_dim = attn_dim - dataset.data_attribute.shape[1]

attn_mask = True
num_heads = 10

# generate discriminators and generator
discriminator = Discriminator(dataset.data_feature_shape, dataset.data_attribute_shape)
attr_discriminator = AttrDiscriminator(dataset.data_attribute_shape)


if gan_type == "RNN":
    generator = DoppelGANgerGeneratorRNN(noise_dim=noise_dim, feature_outputs=dataset.data_feature_outputs,
                                         attribute_outputs=dataset.data_attribute_outputs,
                                         real_attribute_mask=dataset.real_attribute_mask, device=device,
                                         sample_len=sample_len)
else:
    generator = DoppelGANgerGeneratorAttention(noise_dim=noise_dim, feature_outputs=dataset.data_feature_outputs,
                                               attribute_outputs=dataset.data_attribute_outputs,
                                               real_attribute_mask=dataset.real_attribute_mask, device=device,
                                               sample_len=sample_len, num_heads=num_heads, attn_dim=attn_dim)

# define optimizer
g_lr = 0.0001
g_beta1 = 0.5
d_lr = 0.0001
d_beta1 = 0.5
attr_d_lr = 0.0001
attr_d_beta1 = 0.5

attr_opt = torch.optim.Adam(discriminator.parameters(), lr=d_lr, betas=(0.5, 0.999))
d_attr_opt = torch.optim.Adam(attr_discriminator.parameters(), lr=attr_d_lr, betas=(0.5, 0.999))
gen_opt = torch.optim.Adam(generator.parameters(), lr=g_lr, betas=(0.5, 0.999))

data_feature_shape = dataset.data_feature_shape
# define Hyperparameters
epoch = 400
d_rounds = 1
g_rounds = 1
d_gp_coe = 10.0
attr_d_gp_coe = 10.0
g_attr_d_coe = 1.0


for n in range(1, 2, 1):
    for i in range(0, 400, 20):
        model_dir = "runs/{}/{}/{}/checkpoint/epoch_{}".format(dataset_name, gan_type, n, i)
        trainer = Trainer(discriminator=discriminator, attr_discriminator=attr_discriminator, generator=generator,
                          criterion=None, dis_optimizer=attr_opt, addi_dis_optimizer=d_attr_opt, gen_optimizer=gen_opt,
                          real_train_dl=None, data_feature_shape=data_feature_shape, device=device,
                          noise_dim=noise_dim,
                          sample_len=sample_len, d_rounds=d_rounds, g_rounds=g_rounds)
        trainer.load(model_dir)

        # start sampling
        # for the start we want to 'produce' as many samples as we have data available
        rounds = dataset.data_attribute_shape[0] // batch_size
        sampled_features = np.zeros((0, dataset.data_feature_shape[1], dataset.data_feature_shape[2] - 2))
        sampled_attributes = np.zeros((0, dataset.data_attribute_shape[1]))
        sampled_gen_flags = np.zeros((0, dataset.data_feature_shape[1]))
        sampled_lengths = np.zeros(0)
        for i in range(rounds):
            real_attribute_input_noise = trainer.gen_attribute_input_noise(batch_size).to(device)
            addi_attribute_input_noise = trainer.gen_attribute_input_noise(batch_size).to(device)
            feature_input_noise = trainer.gen_feature_input_noise(batch_size, trainer.sample_time).to(device)
            features, attributes, gen_flags, lengths = trainer.sample_from(real_attribute_input_noise,
                                                                           addi_attribute_input_noise,
                                                                           feature_input_noise)
            sampled_features = np.concatenate((sampled_features, features), axis=0)
            sampled_attributes = np.concatenate((sampled_attributes, attributes), axis=0)
            sampled_gen_flags = np.concatenate((sampled_gen_flags, gen_flags), axis=0)
            sampled_lengths = np.concatenate((sampled_lengths, lengths), axis=0)
        np.savez("{0}/generated_samples.npz".format(model_dir), sampled_features=sampled_features,
                 sampled_attributes=sampled_attributes, sampled_gen_flags=sampled_gen_flags,
                 sampled_lengths=sampled_lengths)
