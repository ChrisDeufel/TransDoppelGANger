from torch.utils.data import DataLoader
import torch

from data import Data
from trainer import Trainer
from gan.network import Discriminator, AttrDiscriminator, DoppelGANgerGenerator

dataset = "FCC_MBA"
checkpoint_dir = 'runs/FCC_MBA_12/checkpoint'
logging_file = 'runs/FCC_MBA_12/time.log'
sample_len = 8
batch_size = 100
noise_dim = 5
# load data

dataset = Data(sample_len=sample_len, name=dataset)
real_train_dl = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# generate discriminators and generator
discriminator = Discriminator(dataset.data_feature, dataset.data_attribute)
attr_discriminator = AttrDiscriminator(dataset.data_attribute)
generator = DoppelGANgerGenerator(noise_dim=noise_dim, feature_outputs=dataset.data_feature_outputs,
                                  attribute_outputs=dataset.data_attribute_outputs,
                                  real_attribute_mask=dataset.real_attribute_mask, sample_len=sample_len)

# define optimizer
g_lr = 0.001
g_beta1 = 0.5
d_lr = 0.001
d_beta1 = 0.5
attr_d_lr = 0.001
attr_d_beta1 = 0.5

attr_opt = torch.optim.Adam(discriminator.parameters(), lr=d_lr, betas=(0.5, 0.999))
d_attr_opt = torch.optim.Adam(attr_discriminator.parameters(), lr=attr_d_lr, betas=(0.5, 0.999))
gen_opt = torch.optim.Adam(generator.parameters(), lr=g_lr, betas=(0.5, 0.999))

data_feature_shape = dataset.data_feature.shape
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

trainer = Trainer(discriminator=discriminator, attr_discriminator=attr_discriminator, generator=generator,
                  criterion=None, dis_optimizer=attr_opt, addi_dis_optimizer=d_attr_opt, gen_optimizer=gen_opt,
                  real_train_dl=real_train_dl, data_feature_shape=data_feature_shape, checkpoint_dir=checkpoint_dir,
                  logging_file=logging_file, sample_len=sample_len)
# model_dir = "runs/web_17/checkpoint/epoch_395"
# trainer.load(model_dir)
trainer.train(epochs=epoch, writer_frequency=1, saver_frequency=5)
