from torch.utils.data import DataLoader
import torch
import logging
from data import TimeGanData
import os
from trainer_timeGAN_original import TimeGAN
import sys

def main():

    dataset_name = sys.argv[1]
    batch_size = 20
    dataset = TimeGanData(name=dataset_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    gan_type = "Time_GAN"
    checkpoint_dir = 'runs/{}/{}/1/checkpoint'.format(dataset_name, gan_type)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    time_logging_file = 'runs/{}/{}/1/time.log'.format(dataset_name, gan_type)
    config_logging_file = 'runs/{}/{}/1/config.log'.format(dataset_name, gan_type)


    real_train_dl = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    trainer = TimeGAN(real_train_dl, device=device, checkpoint_dir=checkpoint_dir, config_log=config_logging_file,
                      time_log=time_logging_file)
    epochs = 400
    trainer.train(epochs=epochs, saver_frequency=20)

if __name__ == "__main__":
    main()