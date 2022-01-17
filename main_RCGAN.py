from torch.utils.data import DataLoader
import torch
import logging
from data import Data
import os
from trainer_RCGAN import RCGAN

dataset_name = "index_growth_1mo"
batch_size = 28
dataset = Data(name=dataset_name, sample_len=1)
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
gan_type = "RCGAN"
if gan_type == 'RGAN':
    isConditional = False
else:
    isConditional = True
checkpoint_dir = 'runs/{}/{}/test/checkpoint'.format(dataset_name, gan_type)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

time_logging_file = 'runs/{}/{}/test/time.log'.format(dataset_name, gan_type)
config_logging_file = 'runs/{}/{}/test/config.log'.format(dataset_name, gan_type)

real_train_dl = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
trainer = RCGAN(real_train_dl, device=device, checkpoint_dir=checkpoint_dir, time_logging_file=time_logging_file,
                config_logging_file=config_logging_file, isConditional=isConditional)
epochs = 2
trainer.train(epochs=epochs, saver_frequency=1)
