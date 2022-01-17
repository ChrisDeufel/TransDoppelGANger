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

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
dataset = "index_growth_1mo"
gan_type = 'RNN'
dis_type = 'normal'
checkpoint_dir = 'runs/{}/{}_Dis{}/test/checkpoint'.format(dataset, gan_type, dis_type)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
time_logging_file = 'runs/{}/{}_Dis{}/test/time.log'.format(dataset, gan_type, dis_type)
config_logging_file = 'runs/{}/{}_Dis{}/test/config.log'.format(dataset, gan_type, dis_type)
# SET UP LOGGING
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# set up time handler
time_formatter = logging.Formatter('%(asctime)s:%(message)s')
time_handler = logging.FileHandler(time_logging_file)
time_handler.setLevel(logging.INFO)
time_handler.setFormatter(time_formatter)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(time_formatter)
add_handler_trainer([time_handler, stream_handler])
# setup config handler
config_formatter = logging.Formatter('%(message)s')
config_handler = logging.FileHandler(config_logging_file)
config_handler.setLevel(logging.INFO)
config_handler.setFormatter(config_formatter)
logger.addHandler(config_handler)

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
logger.info("Sample Length: {0}".format(sample_len))
logger.info("Batch Size: {0}".format(batch_size))
logger.info("Noise Dimension: {0}".format(noise_dim))
logger.info("Attention Mask: {0}".format(attn_mask))
logger.info("Number of Attention Heads: {0}".format(num_heads))

# generate discriminators and generator
# discriminator = Discriminator(dataset.data_feature_shape, dataset.data_attribute_shape)
# logger.info("DISCRIMINATOR: {0}".format(discriminator))
# attr_discriminator = AttrDiscriminator(dataset.data_attribute_shape)
# logger.info("ATTRIBUTE DISCRIMINATOR: {0}".format(attr_discriminator))

# if gan_type == "RNN":
#     generator = DoppelGANgerGeneratorRNN(noise_dim=noise_dim, feature_outputs=dataset.data_feature_outputs,
#                                          attribute_outputs=dataset.data_attribute_outputs,
#                                          real_attribute_mask=dataset.real_attribute_mask, device=device,
#                                          sample_len=sample_len)
# else:
#     generator = DoppelGANgerGeneratorAttention(noise_dim=noise_dim, feature_outputs=dataset.data_feature_outputs,
#                                                attribute_outputs=dataset.data_attribute_outputs,
#                                                real_attribute_mask=dataset.real_attribute_mask, device=device,
#                                                sample_len=sample_len, num_heads=num_heads, attn_dim=attn_dim)

# logger.info("GENERATOR: {0}".format(generator))
# define optimizer
g_lr = 0.001
g_beta1 = 0.5
logger.info("g_lr: {0} / g_beta1: {1}".format(g_lr, g_beta1))
d_lr = 0.001
d_beta1 = 0.5
logger.info("d_lr: {0} / d_beta1: {1}".format(d_lr, d_beta1))
attr_d_lr = 0.001
attr_d_beta1 = 0.5
logger.info("attr_d_lr: {0} / attr_d_beta1: {1}".format(attr_d_lr, attr_d_beta1))

data_feature_shape = dataset.data_feature_shape
# define Hyperparameters
epoch = 400
d_rounds = 1
logger.info("d_rounds: {0}".format(d_rounds))
g_rounds = 1
logger.info("g_rounds: {0}".format(g_rounds))
d_gp_coe = 10.0
logger.info("d_gp_coefficient: {0}".format(d_gp_coe))
attr_d_gp_coe = 10.0
logger.info("attr_d_gp_coefficient: {0}".format(attr_d_gp_coe))
g_attr_d_coe = 1.0
logger.info("g_attr_d_coe: {0}".format(g_attr_d_coe))

trainer = Trainer(criterion=None, real_train_dl=real_train_dl, data_feature_shape=data_feature_shape, device=device,
                  checkpoint_dir=checkpoint_dir, noise_dim=noise_dim,
                  sample_len=sample_len, d_rounds=d_rounds, g_rounds=g_rounds, gen_type=gan_type, dis_type=dis_type,
                  att_dim=attn_dim, num_heads=num_heads, g_lr=g_lr, g_beta1=g_beta1, d_lr=d_lr,
                  d_beta1=d_beta1, attr_d_lr=attr_d_lr, attr_d_beta1=attr_d_beta1)
trainer.train(epochs=epoch, writer_frequency=1, saver_frequency=20)
