from torch.utils.data import DataLoader
import torch
import numpy as np
from trainer import Trainer
from trainer_RCGAN import RCGAN
from trainer_timeGAN import TimeGAN
from trainer_CGAN import CGAN
from data import Data, LargeData, SplitData, TimeGanData

device = "cuda" if torch.cuda.is_available() else "cpu"
dataset_name = "index_growth_1mo"
gan_type = 'Time_GAN'
sample_len = 1
batch_size = 20
gen_flag = True

# load data
if dataset_name == "transactions":
    dataset = SplitData(sample_len, name=dataset_name)
else:
    if gan_type == "Time_GAN":
        dataset = TimeGanData(name=dataset_name)
    else:
        dataset = Data(sample_len=sample_len, name=dataset_name)
real_train_dl = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# GENERAL
lr = 0.001
beta1 = 0.5
noise_dim = 5
# FOR RNN OR TRANSFORMER
attn_dim = 100
attn_mask = True
num_heads = 5
g_lr = 0.0001
g_beta1 = 0.5
d_lr = 0.0001
d_beta1 = 0.5
attr_d_lr = 0.0001
attr_d_beta1 = 0.5

# SPECIFICALLY FOR RCGAN
noise_size = noise_dim
hidden_size_gen = 100
num_layer_gen = 1
hidden_size_dis = 100
num_layer_dis = 1
checkpoint_dir = ""
isConditional = False

# SPECIFICALLY FOR TIME GAN
z_dim = 6
hidden_dim = 24
num_layer = 3
istrain = True
w_gamma = 1
w_es = 0.1
w_e0 = 10
w_g = 100

# SPECIFICALLY FOR CGAN


data_feature_shape = dataset.data_feature_shape
# define Hyperparameters
epoch = 400

for n in range(1, 2, 1):
    for i in range(0, 400, 20):
        model_dir = "runs/{}/{}/{}/checkpoint/epoch_{}".format(dataset_name, gan_type, n, i)
        if gan_type == "RNN" or gan_type == "TRANSFORMER":
            trainer = Trainer(criterion=None,
                              real_train_dl=real_train_dl, data_feature_shape=data_feature_shape, device=device,
                              noise_dim=noise_dim, sample_len=sample_len, gen_type=gan_type, att_dim=attn_dim,
                              num_heads=num_heads, g_lr=g_lr, g_beta1=g_beta1, d_lr=d_lr, d_beta1=d_beta1,
                              attr_d_lr=attr_d_lr, attr_d_beta1=attr_d_beta1
                              )
        elif gan_type == "RCGAN" or gan_type == "RGAN":
            trainer = RCGAN(real_train_dl, device=device, checkpoint_dir=checkpoint_dir, isConditional=isConditional)
        elif gan_type == "CGAN":
            trainer = CGAN(real_train_dl, device=device, checkpoint_dir=checkpoint_dir)
        else:
            trainer = TimeGAN(real_train_dl, device=device, checkpoint_dir=checkpoint_dir, config_log=None,
                              time_log=None)
        trainer.load(model_dir)

        # start sampling
        # for the start we want to 'produce' as many samples as we have data available
        while dataset.data_attribute_shape[0] % batch_size != 0:
            batch_size -= 1
        rounds = dataset.data_attribute_shape[0] // batch_size
        sampled_features = np.zeros((0, dataset.data_feature_shape[1], dataset.data_feature_shape[2]-2))
        sampled_attributes = np.zeros((0, dataset.data_attribute_shape[1]))
        sampled_gen_flags = np.zeros((0, dataset.data_feature_shape[1]))
        sampled_lengths = np.zeros(0)
        for i in range(rounds):
            features, attributes, gen_flags, lengths = trainer.sample_from(batch_size=batch_size)
            sampled_features = np.concatenate((sampled_features, features), axis=0)
            sampled_attributes = np.concatenate((sampled_attributes, attributes), axis=0)
            sampled_gen_flags = np.concatenate((sampled_gen_flags, gen_flags), axis=0)
            sampled_lengths = np.concatenate((sampled_lengths, lengths), axis=0)
        np.savez("{0}/generated_samples.npz".format(model_dir), sampled_features=sampled_features,
                 sampled_attributes=sampled_attributes, sampled_gen_flags=sampled_gen_flags,
                 sampled_lengths=sampled_lengths)
