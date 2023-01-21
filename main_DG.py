from torch.utils.data import DataLoader
import torch
from data import Data
from trainer import DoppelGANger, CGAN, RCGAN, NAIVEGAN, TimeGAN, TimeGAN2
import os
from util import options_parser



def main():
    parser = options_parser()
    args = parser.parse_args()
    device = args.device
    dataset = args.dataset
    gan_type = args.gan_type
    dis_type = args.dis_type
    wl = args.w_lambert == 'True'
    ks = None if args.kernel_smoothing is None else args.kernel_smoothing

    checkpoint_dir = 'runs/{}'.format(dataset)
    if wl:
        checkpoint_dir = "{}_wl".format(checkpoint_dir)
    if ks is not None:
        checkpoint_dir = "{}_ks_{}".format(checkpoint_dir, ks)
    if args.dis_type is None:
        checkpoint_dir = '{}/{}/1'.format(checkpoint_dir, gan_type)
    else:
        checkpoint_dir = '{}/Gen_{}_Dis_{}/1'.format(checkpoint_dir, gan_type, dis_type)

    time_logging_file = '{}/time.log'.format(checkpoint_dir)
    config_logging_file = '{}/config.log'.format(checkpoint_dir)
    checkpoint_dir = '{}/checkpoint'.format(checkpoint_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    sample_len = int(args.sample_len)

    batch_size = args.batch_size
    # load data
    dataset = Data(sample_len=sample_len, name=dataset, w_lambert=wl, ks=ks)
    real_train_dl = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    # define Hyperparameters
    epoch = args.num_epochs
    save_frequency = args.save_frequency
    if gan_type == 'RCGAN' or gan_type == 'RGAN':
        is_conditional = args.is_conditional
        trainer = RCGAN(real_train_dl, device=device, checkpoint_dir=checkpoint_dir,
                        time_logging_file=time_logging_file, batch_size=batch_size,
                        config_logging_file=config_logging_file, isConditional=is_conditional)
    elif gan_type == 'NaiveGAN':
        trainer = NAIVEGAN(real_train_dl, device=device, checkpoint_dir=checkpoint_dir, batch_size=batch_size,
                           time_logging_file=time_logging_file,
                           config_logging_file=config_logging_file)
    elif gan_type == 'CGAN':
        trainer = CGAN(real_train_dl, device=device, batch_size=batch_size, checkpoint_dir=checkpoint_dir,
                       time_logging_file=time_logging_file, config_logging_file=config_logging_file)
    elif gan_type == 'TimeGAN':
        trainer = TimeGAN2(real_train_dl, device=device, checkpoint_dir=checkpoint_dir, batch_size=batch_size,
                           config_logging_file=config_logging_file,
                           time_logging_file=time_logging_file)
    else:
        trainer = DoppelGANger(real_train_dl=real_train_dl, device=device,
                               checkpoint_dir=checkpoint_dir, time_logging_file=time_logging_file,
                               config_logging_file=config_logging_file, sample_len=sample_len, batch_size=batch_size,
                               gen_type=gan_type, dis_type=dis_type)

    trainer.train(epochs=epoch, writer_frequency=1, saver_frequency=save_frequency)


if __name__ == "__main__":
    main()
