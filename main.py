from torch.utils.data import DataLoader
import torch
import logging
from load_data import load_data
from data import Data, LargeData
from trainer import Trainer, add_handler_trainer
# from gan.network import Discriminator, AttrDiscriminator, DoppelGANgerGenerator
from gan.network_2 import Discriminator, AttrDiscriminator, DoppelGANgerGeneratorAttention, DoppelGANgerGeneratorRNN
from util import add_gen_flag, normalize_per_sample
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
#device = 'cpu'
dataset = "FCC_MBA"
checkpoint_dir = 'runs/{0}/attention_test/checkpoint'.format(dataset)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
time_logging_file = 'runs/{0}/attention_test/time.log'.format(dataset)
config_logging_file = 'runs/{0}/attention_test/config.log'.format(dataset)
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

sample_len = 8
batch_size = 100
attn_dim = 100
# load data
dataset = Data(sample_len=sample_len, name=dataset)
real_train_dl = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

noise_dim = attn_dim-dataset.data_attribute.shape[1]
attn_mask = True
num_heads = 5
logger.info("Sample Length: {0}".format(sample_len))
logger.info("Batch Size: {0}".format(batch_size))
logger.info("Noise Dimension: {0}".format(noise_dim))
logger.info("Attention Mask: {0}".format(attn_mask))
logger.info("Number of Attention Heads: {0}".format(num_heads))

# generate discriminators and generator
discriminator = Discriminator(dataset.data_feature, dataset.data_attribute)
logger.info("DISCRIMINATOR: {0}".format(discriminator))
attr_discriminator = AttrDiscriminator(dataset.data_attribute)
logger.info("ATTRIBUTE DISCRIMINATOR: {0}".format(attr_discriminator))
generator = DoppelGANgerGeneratorAttention(noise_dim=noise_dim, feature_outputs=dataset.data_feature_outputs,
                                           attribute_outputs=dataset.data_attribute_outputs,
                                           real_attribute_mask=dataset.real_attribute_mask, device=device,
                                           sample_len=sample_len, num_heads=num_heads, attn_dim=attn_dim)
logger.info("GENERATOR: {0}".format(generator))
# define optimizer
g_lr = 0.0001
g_beta1 = 0.5
logger.info("g_lr: {0} / g_beta1: {1}".format(g_lr, g_beta1))
d_lr = 0.0001
d_beta1 = 0.5
logger.info("d_lr: {0} / d_beta1: {1}".format(d_lr, d_beta1))
attr_d_lr = 0.0001
attr_d_beta1 = 0.5
logger.info("attr_d_lr: {0} / attr_d_beta1: {1}".format(attr_d_lr, attr_d_beta1))

attr_opt = torch.optim.Adam(discriminator.parameters(), lr=d_lr, betas=(0.5, 0.999))
d_attr_opt = torch.optim.Adam(attr_discriminator.parameters(), lr=attr_d_lr, betas=(0.5, 0.999))
gen_opt = torch.optim.Adam(generator.parameters(), lr=g_lr, betas=(0.5, 0.999))

data_feature_shape = dataset.data_feature.shape
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

trainer = Trainer(discriminator=discriminator, attr_discriminator=attr_discriminator, generator=generator,
                  criterion=None, dis_optimizer=attr_opt, addi_dis_optimizer=d_attr_opt, gen_optimizer=gen_opt,
                  real_train_dl=real_train_dl, data_feature_shape=data_feature_shape, device=device,
                  checkpoint_dir=checkpoint_dir, noise_dim=noise_dim,
                  logging_file=time_logging_file, sample_len=sample_len, d_rounds=d_rounds, g_rounds=g_rounds)
trainer.train(epochs=epoch, writer_frequency=1, saver_frequency=10)
