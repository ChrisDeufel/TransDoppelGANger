from torch.utils.data import DataLoader
import torch
import logging
from load_data import load_data
from data import Data, LargeData, SplitData
from trainer import Trainer
from gan.network import Discriminator, AttrDiscriminator, DoppelGANgerGeneratorAttention, DoppelGANgerGeneratorRNN
from util import add_gen_flag, normalize_per_sample
import os
import numpy as np

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
dataset = "FCC_MBA"
gan_type = 'RNN'
dis_type = 'normal'
checkpoint_dir = 'runs/{}/Gen_{}_Dis_{}/test/checkpoint'.format(dataset, gan_type, dis_type)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
time_logging_file = 'runs/{}/Gen_{}_Dis_{}/test/time.log'.format(dataset, gan_type, dis_type)
config_logging_file = 'runs/{}/Gen_{}_Dis_{}/test/config.log'.format(dataset, gan_type, dis_type)

sample_len = 1
batch_size = 100
attn_dim = 100
# load data
if dataset == "transactions":
    dataset = SplitData(sample_len, name=dataset)
else:
    dataset = Data(sample_len=sample_len, name=dataset)
real_train_dl = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
noise_dim = 5

attn_mask = True
num_heads = 10
# logger.info("GENERATOR: {0}".format(generator))
# define optimizer
g_lr = 0.001
g_beta1 = 0.5
d_lr = 0.001
d_beta1 = 0.5
attr_d_lr = 0.001
attr_d_beta1 = 0.5

data_feature_shape = dataset.data_feature_shape
# define Hyperparameters
epoch = 400
d_rounds = 1
g_rounds = 1
d_gp_coe = 10.0
attr_d_gp_coe = 10.0
g_attr_d_coe = 1.0

trainer = Trainer(real_train_dl=real_train_dl, device=device,
                  checkpoint_dir=checkpoint_dir, time_logging_file=time_logging_file,
                  config_logging_file=config_logging_file, noise_dim=noise_dim,
                  sample_len=sample_len, batch_size=batch_size, d_rounds=d_rounds, g_rounds=g_rounds, gen_type=gan_type,
                  dis_type=dis_type,
                  att_dim=attn_dim, num_heads=num_heads, g_lr=g_lr, g_beta1=g_beta1, d_lr=d_lr,
                  d_beta1=d_beta1, attr_d_lr=attr_d_lr, attr_d_beta1=attr_d_beta1)
trainer.train(epochs=epoch, writer_frequency=1, saver_frequency=20)
