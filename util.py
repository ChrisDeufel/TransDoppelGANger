from output import OutputType, Output, Normalization
import numpy as np
import matplotlib
import os
import pickle
from torch.distributions.multivariate_normal import MultivariateNormal
import torch
from sklearn import metrics
from scipy.special import lambertw
#from mpmath import lambertw

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging
import argparse

def evaluation(sampled_data, dataset, epoch):
    data = []
    # append real data
    data.append({
        'data_feature': dataset.data_feature,
        'data_attribute': dataset.data_attribute,
        'data_gen_flag': dataset.data_gen_flag,
        'color': 'yellow',
        'name': 'REAL'
    })
    data.append({
        'data_feature': sampled_data['sampled_features'],
        'data_attribute': sampled_data['sampled_attributes'],
        'data_gen_flag': sampled_data['sampled_gen_flags'],
        'data_lengths': sampled_data['sampled_lengths'],
        'color': 'blue',
        'name': gan_type
    })

def add_handler(logger, handlers):
    for handler in handlers:
        logger.addHandler(handler)


def setup_logging(time_logging_file, config_logging_file):
    # SET UP LOGGING
    config_logger = logging.getLogger("config_logger")
    config_logger.setLevel(logging.INFO)
    # config_logger.setLevel(logging.INFO)
    time_logger = logging.getLogger("time_logger")
    time_logger.setLevel(logging.INFO)
    # time_logger.setLevel(logging.INFO)
    # set up time handler
    time_formatter = logging.Formatter('%(asctime)s:%(message)s')
    time_handler = logging.FileHandler(time_logging_file)
    time_handler.setLevel(logging.INFO)
    time_handler.setFormatter(time_formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(time_formatter)
    add_handler(time_logger, [time_handler, stream_handler])
    # setup config handler
    config_formatter = logging.Formatter('%(message)s')
    config_handler = logging.FileHandler(config_logging_file)
    config_handler.setLevel(logging.INFO)
    config_handler.setFormatter(config_formatter)
    config_logger.addHandler(config_handler)
    return config_logger, time_logger


def options_parser():
    parser = argparse.ArgumentParser(description='Train a GAN to generate sequential, real-valued data.')
    parser.add_argument('-da', '--dataset', help='name of dataset', type=str, default='FCC_MBA')
    parser.add_argument('-wl', '--w_lambert', help='data should be w lambert transformed', type=bool, default=False)
    parser.add_argument('-ks', '--kernel_smoothing', help='window size for kernel smoothing', type=int, default=None)
    parser.add_argument('-gt', '--gan_type', help='name of GAN', type=str, default='RNN')
    parser.add_argument('-dt', '--dis_type', help='discriminator type', type=str, default=None)
    parser.add_argument('-sl', '--sample_len', help='sample length for DoppelGANger', type=int, default=1)
    parser.add_argument('-bs', '--batch_size', help='Minibatch size', type=int, default=20)
    parser.add_argument('-nd', '--noise_dim', help='Dimension of random Input Noise', type=int, default=20)
    parser.add_argument('-ep', '--num_epochs', help='Number of Epochs to train', type=int, default=401)
    parser.add_argument('-ic', '--is_conditional', help='For RC GAN', type=bool, default=False)
    parser.add_argument('-dv', '--device', help='Run on GPU or CPU', type=str, default='cpu')
    parser.add_argument('-sf', '--save_frequency', help='every x epoch to save model parameters', type=int, default=20)
    return parser


def setup_logger(name, log_file, formatter, level=logging.INFO):
    """To setup as many loggers as you want"""
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


def draw_attribute(data, outputs, path=None):
    if isinstance(data, list):
        num_sample = len(data)
    else:
        num_sample = data.shape[0]
    id_ = 0
    for i in range(len(outputs)):
        if outputs[i].type_ == OutputType.CONTINUOUS:
            for j in range(outputs[i].dim):
                plt.figure()
                for k in range(num_sample):
                    plt.scatter(
                        k,
                        data[k][id_],
                        s=12)
                if path is None:
                    plt.show()
                else:
                    plt.savefig("{},output-{},dim-{}.png".format(path, i, j))
                plt.xlabel("sample")
                plt.close()
                id_ += 1
        elif outputs[i].type_ == OutputType.DISCRETE:
            plt.figure()
            for j in range(num_sample):
                plt.scatter(
                    j,
                    np.argmax(data[j][id_: id_ + outputs[i].dim],
                              axis=0),
                    s=12)
            plt.xlabel("sample")
            if path is None:
                plt.show()
            else:
                plt.savefig("{},output-{}.png".format(path, i))
            plt.close()
            id_ += outputs[i].dim
        else:
            raise Exception("unknown output type")


def draw_feature(data, lengths, outputs, path=None):
    if isinstance(data, list):
        num_sample = len(data)
    else:
        num_sample = data.shape[0]
    id_ = 0
    for i in range(len(outputs)):
        if outputs[i].type_ == OutputType.CONTINUOUS:
            for j in range(outputs[i].dim):
                plt.figure()
                for k in range(num_sample):
                    plt.plot(
                        range(int(lengths[k])),
                        data[k][:int(lengths[k]), id_],
                        "o-",
                        markersize=3,
                        label="sample-{}".format(k))
                plt.legend()
                if path is None:
                    plt.show()
                else:
                    plt.savefig("{},output-{},dim-{}.png".format(path, i, j))
                plt.close()
                id_ += 1
        elif outputs[i].type_ == OutputType.DISCRETE:
            plt.figure()
            for j in range(num_sample):
                plt.plot(
                    range(int(lengths[j])),
                    np.argmax(data[j][:int(lengths[j]),
                              id_: id_ + outputs[i].dim],
                              axis=1),
                    "o-",
                    markersize=3,
                    label="sample-{}".format(j))

            plt.legend()
            if path is None:
                plt.show()
            else:
                plt.savefig("{},output-{}.png".format(path, i))
            plt.close()
            id_ += outputs[i].dim
        else:
            raise Exception("unknown output type")


def renormalize_per_sample(data_feature, data_attribute, data_feature_outputs,
                           data_attribute_outputs, gen_flags,
                           num_real_attribute):
    attr_dim = 0
    for i in range(num_real_attribute):
        attr_dim += data_attribute_outputs[i].dim
    attr_dim_cp = attr_dim

    fea_dim = 0
    for output in data_feature_outputs:
        if output.type_ == OutputType.CONTINUOUS:
            for _ in range(output.dim):
                max_plus_min_d_2 = data_attribute[:, attr_dim]
                max_minus_min_d_2 = data_attribute[:, attr_dim + 1]
                attr_dim += 2

                max_ = max_plus_min_d_2 + max_minus_min_d_2
                min_ = max_plus_min_d_2 - max_minus_min_d_2

                max_ = np.expand_dims(max_, axis=1)
                min_ = np.expand_dims(min_, axis=1)

                if output.normalization == Normalization.MINUSONE_ONE:
                    data_feature[:, :, fea_dim] = \
                        (data_feature[:, :, fea_dim] + 1.0) / 2.0

                data_feature[:, :, fea_dim] = \
                    data_feature[:, :, fea_dim] * (max_ - min_) + min_

                fea_dim += 1
        else:
            fea_dim += output.dim

    tmp_gen_flags = np.expand_dims(gen_flags, axis=2)
    data_feature = data_feature * tmp_gen_flags

    data_attribute = data_attribute[:, 0: attr_dim_cp]

    return data_feature, data_attribute


def kernel_smoothing(x, ks):
    y = np.zeros_like(x)
    for i in range(len(x)):
        if i - int(ks / 2) < 0:
            y[i] = np.mean(x[:ks])
        elif i + int(ks / 2) > (len(x) - 1):
            y[i] = np.mean(x[-ks:])
        else:
            y[i] = np.mean(x[i - int(ks / 2):i + int(ks / 2)])
    return y


def normalize_per_sample(data_feature, data_attribute, data_feature_outputs,
                         data_attribute_outputs, data_gen_flag, eps=0.0001, w_lambert=True, ks=None):
    # assume all samples have maximum length
    data_feature_min = np.zeros((data_feature.shape[0], data_feature.shape[2]))
    data_feature_max = np.zeros((data_feature.shape[0], data_feature.shape[2]))
    for i in range(len(data_feature)):
        len_feature = np.count_nonzero(data_gen_flag[i, :])
        data_feature_min[i, :] = np.amin(data_feature[i, :len_feature, :], axis=0)
        data_feature_max[i, :] = np.amax(data_feature[i, :len_feature, :], axis=0)
    #data_feature_min = np.amin(data_feature, axis=1)
    #data_feature_max = np.amax(data_feature, axis=1)

    additional_attribute = []
    additional_attribute_outputs = []

    dim = 0
    for output in data_feature_outputs:
        if output.type_ == OutputType.CONTINUOUS:
            for _ in range(output.dim):
                max_ = data_feature_max[:, dim] + eps
                min_ = data_feature_min[:, dim] - eps

                additional_attribute.append((max_ + min_) / 2.0)
                additional_attribute.append((max_ - min_) / 2.0)
                if output.normalization == Normalization.MINUSONE_ONE:
                    additional_attribute_outputs.append(Output(
                        type_=OutputType.CONTINUOUS,
                        dim=1,
                        normalization=Normalization.MINUSONE_ONE,
                        is_gen_flag=False))
                else:
                    additional_attribute_outputs.append(Output(
                        type_=OutputType.CONTINUOUS,
                        dim=1,
                        normalization=Normalization.ZERO_ONE,
                        is_gen_flag=False))

                max_ = np.expand_dims(max_, axis=1)
                min_ = np.expand_dims(min_, axis=1)

                data_feature[:, :, dim] = \
                    (data_feature[:, :, dim] - min_) / (max_ - min_)
                if output.normalization == Normalization.MINUSONE_ONE:
                    data_feature[:, :, dim] = \
                        data_feature[:, :, dim] * 2.0 - 1.0

                dim += 1
        else:
            dim += output.dim

    real_attribute_mask = ([True] * len(data_attribute_outputs) +
                           [False] * len(additional_attribute_outputs))

    additional_attribute = np.stack(additional_attribute, axis=1)
    data_attribute = np.concatenate(
        [data_attribute, additional_attribute], axis=1)
    data_attribute_outputs.extend(additional_attribute_outputs)


    if w_lambert:
        data_feature = lambertw(data_feature).real
    if ks is not None:
        for i in range(len(data_feature)):
            for d in range(data_feature.shape[2]):
                data_feature[i, :, d] = kernel_smoothing(data_feature[i, :, d], ks)
    for i in range(len(data_feature)):
        len_feature = np.count_nonzero(data_gen_flag[i, :])
        data_feature[i, len_feature:, :] = 0
    return data_feature, data_attribute, data_attribute_outputs, \
           real_attribute_mask


def normalize_per_sample_split(path, nr_samples, data_feature_outputs, data_attribute_outputs, eps=1e-4):
    # create folder for normalized samples if not exists
    if not os.path.exists("{}/normalized".format(path)):
        os.makedirs("{}/normalized".format(path))
    # assume all samples have maximum length
    additional_attribute_outputs = []

    counter = 0
    dim = 0
    for output in data_feature_outputs:
        if output.type_ == OutputType.CONTINUOUS:
            for _ in range(output.dim):
                for u in range(nr_samples):
                    if counter < nr_samples:
                        attributes = np.load("{}/{}_data_attribute.npy".format(path, u))
                        features = np.load("{}/{}_data_feature.npy".format(path, u))
                        counter += 1
                    else:
                        attributes = np.load("{}/normalized/{}_data_attribute.npy".format(path, u))
                        features = np.load("{}/normalized/{}_data_feature.npy".format(path, u))
                    max_ = np.expand_dims(np.array(np.amax(features, axis=0)[dim] + eps), axis=0)
                    min_ = np.expand_dims(np.array(np.amin(features, axis=0)[dim] - eps), axis=0)
                    attributes = np.concatenate((attributes, (max_ + min_) / 2.0), axis=0)
                    attributes = np.concatenate((attributes, (max_ - min_) / 2.0), axis=0)
                    features[:, dim] = \
                        (features[:, dim] - min_) / (max_ - min_)
                    if output.normalization == Normalization.MINUSONE_ONE:
                        features[:, dim] = \
                            features[:, dim] * 2.0 - 1.0
                    # save normalized sample
                    np.save("{}/normalized/{}_data_attribute.npy".format(path, u), attributes)
                    np.save("{}/normalized/{}_data_feature.npy".format(path, u), features)
                additional_attribute_outputs.append(Output(
                    type_=OutputType.CONTINUOUS,
                    dim=1,
                    normalization=output.normalization,
                    is_gen_flag=False))
                additional_attribute_outputs.append(Output(
                    type_=OutputType.CONTINUOUS,
                    dim=1,
                    normalization=Normalization.ZERO_ONE,
                    is_gen_flag=False))
                dim += 1
        else:
            dim += output.dim
    real_attribute_mask = ([True] * len(data_attribute_outputs) +
                           [False] * len(additional_attribute_outputs))
    np.save('{}/normalized/real_attribute_mask.npy'.format(path), real_attribute_mask)
    data_attribute_outputs.extend(additional_attribute_outputs)
    # drop output files in normalized folder
    dbfile = open('{}/normalized/data_attribute_output.pkl'.format(path), 'ab')
    # source, destination
    pickle.dump(data_attribute_outputs, dbfile)
    dbfile.close()
    # source, destination
    dbfile = open('{}/normalized/data_feature_output.pkl'.format(path), 'ab')
    pickle.dump(data_feature_outputs, dbfile)
    dbfile.close()
    return data_attribute_outputs, real_attribute_mask


def add_gen_flag_split(path, nr_samples, data_gen_flag, data_feature_outputs,
                       sample_len):
    if not os.path.exists("{}/gen_flag".format(path)):
        os.makedirs("{}/gen_flag".format(path))
    for output in data_feature_outputs:
        if output.is_gen_flag:
            raise Exception("is_gen_flag should be False for all"
                            "feature_outputs")

    if len(data_gen_flag.shape) != 2:
        raise Exception("data_gen_flag should be 2 dimension")

    num_sample, length = data_gen_flag.shape

    data_gen_flag = np.expand_dims(data_gen_flag, 2)

    data_feature_outputs.append(Output(
        type_=OutputType.DISCRETE,
        dim=2,
        is_gen_flag=True))

    shift_gen_flag = np.concatenate(
        [data_gen_flag[:, 1:, :],
         np.zeros((data_gen_flag.shape[0], 1, 1))],
        axis=1)
    if length % sample_len != 0:
        raise Exception("length must be a multiple of sample_len")
    data_gen_flag_t = np.reshape(
        data_gen_flag,
        [num_sample, int(length / sample_len), sample_len])
    data_gen_flag_t = np.sum(data_gen_flag_t, 2)
    data_gen_flag_t = data_gen_flag_t > 0.5
    data_gen_flag_t = np.repeat(data_gen_flag_t, sample_len, axis=1)
    data_gen_flag_t = np.expand_dims(data_gen_flag_t, 2)

    for u in range(nr_samples):
        features = np.load("{}/{}_data_feature.npy".format(path, u))
        features = np.concatenate(
            [features, shift_gen_flag[u], (1 - shift_gen_flag[u]) * data_gen_flag_t[u]],
            axis=1)
        np.save("{}/gen_flag/{}_data_feature.npy".format(path, u), features)
    dbfile = open('{}/gen_flag/data_feature_output.pkl'.format(path), 'ab')
    pickle.dump(data_feature_outputs, dbfile)
    dbfile.close()
    return data_feature_outputs


def add_gen_flag(data_feature, data_gen_flag, data_feature_outputs,
                 sample_len):
    for output in data_feature_outputs:
        if output.is_gen_flag:
            raise Exception("is_gen_flag should be False for all"
                            "feature_outputs")

    if (data_feature.shape[2] !=
            np.sum([t.dim for t in data_feature_outputs])):
        raise Exception("feature dimension does not match feature_outputs")

    if len(data_gen_flag.shape) != 2:
        raise Exception("data_gen_flag should be 2 dimension")

    num_sample, length = data_gen_flag.shape

    data_gen_flag = np.expand_dims(data_gen_flag, 2)

    data_feature_outputs.append(Output(
        type_=OutputType.DISCRETE,
        dim=2,
        is_gen_flag=True))

    shift_gen_flag = np.concatenate(
        [data_gen_flag[:, 1:, :],
         np.zeros((data_gen_flag.shape[0], 1, 1))],
        axis=1)
    if length % sample_len != 0:
        raise Exception("length must be a multiple of sample_len")
    data_gen_flag_t = np.reshape(
        data_gen_flag,
        [num_sample, int(length / sample_len), sample_len])
    data_gen_flag_t = np.sum(data_gen_flag_t, 2)
    data_gen_flag_t = data_gen_flag_t > 0.5
    data_gen_flag_t = np.repeat(data_gen_flag_t, sample_len, axis=1)
    data_gen_flag_t = np.expand_dims(data_gen_flag_t, 2)
    data_feature = np.concatenate(
        [data_feature,
         shift_gen_flag,
         (1 - shift_gen_flag) * data_gen_flag_t],
        axis=2)

    return data_feature, data_feature_outputs


def extract_len(data_gen_flag):
    """Returns Maximum sequence length and each sequence length.

    Args:
      - data: original data

    Returns:
      - time: extracted time information
      - max_seq_len: maximum sequence length
    """
    time = list()
    max_seq_len = data_gen_flag.shape[1]
    for i in range(len(data_gen_flag)):
        seq_len = np.count_nonzero(data_gen_flag[i, :])
        time.append(seq_len)

    return np.asarray(time), max_seq_len


def norm_min_max(data):
    """Min-Max Normalizer.

    Args:
      - data: raw data

    Returns:
      - norm_data: normalized data
      - min_val: minimum values (for renormalization)
      - max_val: maximum values (for renormalization)
    """
    min_val = np.min(np.min(data, axis=0), axis=0)
    data = data - min_val  # [3661, 24, 6]

    max_val = np.max(np.max(data, axis=0), axis=0)
    norm_data = data / (max_val + 1e-7)

    return norm_data, min_val, max_val


def random_generator(batch_size, z_dim, T_mb, max_seq_len):
    """Random vector generation.

    Args:
      - batch_size: size of the random vector
      - z_dim: dimension of random vector
      - T_mb: time information for the random vector
      - max_seq_len: maximum sequence length

    Returns:
      - Z_mb: generated random vector
    """
    Z_mb = np.random.uniform(0, 1, (batch_size, max_seq_len, z_dim))
    # Z_mb = list()
    # for i in range(batch_size):
    #     temp = np.zeros([max_seq_len, z_dim])
    #     temp_Z = np.random.uniform(0., 1, [T_mb[i], z_dim])
    #     temp[:T_mb[i], :] = temp_Z
    #     Z_mb.append(temp_Z)
    return Z_mb


# https://github.com/jindongwang/transferlearning/blob/master/code/distance/mmd_numpy_sklearn.py
def calculate_mmd_rbf(X, Y, gamma=1.0):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})
    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()