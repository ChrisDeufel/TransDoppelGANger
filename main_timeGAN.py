from torch.utils.data import DataLoader
import torch
import logging
from data import Data
import os
from trainer_timeGAN import TimeGAN
import sys


def main():
    dataset_name = "index_growth_1mo"
    batch_size = 128
    dataset = Data(name=dataset_name, sample_len=1)
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    gan_type = "Time_GAN"
    checkpoint_dir = 'runs/{}/{}/1/checkpoint'.format(dataset_name, gan_type)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    time_logging_file = 'runs/{}/{}/1/time.log'.format(dataset_name, gan_type)
    config_logging_file = 'runs/{}/{}/1/config.log'.format(dataset_name, gan_type)

    real_train_dl = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    print(len(real_train_dl))
    trainer = TimeGAN(real_train_dl, device=device, checkpoint_dir=checkpoint_dir,
                      config_logging_file=config_logging_file,
                      time_logging_file=time_logging_file)
    epochs = 1
    trainer.train(epochs=epochs, saver_frequency=1)


if __name__ == "__main__":
    main()
