from torch.utils.data import DataLoader
import torch
import logging
from load_data import load_data
from data import Data, LargeData, SplitData
from trainer import Trainer, add_handler_trainer
from gan.network import Discriminator, AttrDiscriminator, DoppelGANgerGeneratorAttention, DoppelGANgerGeneratorRNN
from util import add_gen_flag, normalize_per_sample
import os
import numpy as np

#device = "cuda" if torch.cuda.is_available() else "cpu"
device = 'cpu'
dataset_name = "transactions"
gan_type = "RNN"
sample_len = 100
batch_size = 100
attn_dim = 100

dataset = SplitData(sample_len, name=dataset_name)
real_train_dl = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# noise_dim = attn_dim - dataset.data_attribute.shape[1]
noise_dim = 5
attn_mask = True
num_heads = 10

# generate discriminators and generator
discriminator = Discriminator(dataset.data_feature_shape, dataset.data_attribute_shape)
attr_discriminator = AttrDiscriminator(dataset.data_attribute_shape)
# generator = DoppelGANgerGeneratorAttention(noise_dim=noise_dim, feature_outputs=dataset.data_feature_outputs,
#                                            attribute_outputs=dataset.data_attribute_outputs,
#                                            real_attribute_mask=dataset.real_attribute_mask, device=device,
#                                            sample_len=sample_len, num_heads=num_heads, attn_dim=attn_dim)
generator = DoppelGANgerGeneratorRNN(noise_dim=noise_dim, feature_outputs=dataset.data_feature_outputs,
                                     attribute_outputs=dataset.data_attribute_outputs,
                                     real_attribute_mask=dataset.real_attribute_mask, device=device,
                                     sample_len=sample_len)

# define optimizer
g_lr = 0.001
g_beta1 = 0.5
d_lr = 0.001
d_beta1 = 0.5
attr_d_lr = 0.001
attr_d_beta1 = 0.5
attr_opt = torch.optim.Adam(discriminator.parameters(), lr=d_lr, betas=(d_beta1, d_beta1))
d_attr_opt = torch.optim.Adam(attr_discriminator.parameters(), lr=attr_d_lr, betas=(attr_d_beta1, attr_d_beta1))
gen_opt = torch.optim.Adam(generator.parameters(), lr=g_lr, betas=(g_beta1, g_beta1))
data_feature_shape = dataset.data_feature_shape
# define Hyperparameters
epoch = 400
vis_freq = 500
vis_num_sample = 5
d_rounds = 1
g_rounds = 1
d_gp_coe = 10.0
attr_d_gp_coe = 10.0
g_attr_d_coe = 1.0
extra_checkpoint_freq = 5
num_packing = 1

model_dir = "runs/{}/{}/1/checkpoint/epoch_400".format(dataset_name, gan_type)

trainer = Trainer(discriminator=discriminator, attr_discriminator=attr_discriminator, generator=generator,
                  criterion=None, dis_optimizer=attr_opt, addi_dis_optimizer=d_attr_opt, gen_optimizer=gen_opt,
                  real_train_dl=None, data_feature_shape=data_feature_shape, device=device,
                  noise_dim=noise_dim,
                  sample_len=sample_len, d_rounds=d_rounds, g_rounds=g_rounds)
trainer.load(model_dir)

rounds = dataset.data_attribute_shape[0] // batch_size
sampled_gen_flags = np.zeros((0, dataset.data_feature_shape[1]))
sampled_lengths = np.zeros(0)
idx_counter = 0
for i in range(rounds):
    real_attribute_input_noise = trainer.gen_attribute_input_noise(batch_size).to(device)
    addi_attribute_input_noise = trainer.gen_attribute_input_noise(batch_size).to(device)
    feature_input_noise = trainer.gen_feature_input_noise(batch_size, trainer.sample_time).to(device)
    features, attributes, gen_flags, lengths = trainer.sample_from(real_attribute_input_noise,
                                                                   addi_attribute_input_noise,
                                                                   feature_input_noise)

    for idx in range(len(features)):
        np.save("{}/{}_data_feature.npy".format(model_dir, idx_counter), features[idx, :, :])
        np.save("{}/{}_data_attribute.npy".format(model_dir, idx_counter), features[idx, :])
        idx_counter += 1
    sampled_gen_flags = np.concatenate((sampled_gen_flags, gen_flags), axis=0)
    sampled_lengths = np.concatenate((sampled_lengths, lengths), axis=0)
np.save("{0}/sampled_gen_flags.npy".format(model_dir), sampled_gen_flags)
np.save("{0}/sampled_lengths.npy".format(model_dir), sampled_lengths)
