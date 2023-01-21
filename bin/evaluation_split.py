import math
import statistics
import random
from scipy import stats
from sklearn import metrics
import pandas as pd
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import statsmodels.api as sm
import output
from load_data import load_data
from util import add_gen_flag, normalize_per_sample

sys.modules["output"] = output


######################## AUTOCORRELATION ########################

def autocorr(x):
    nlags = len(x)
    autocorr = sm.tsa.acf(x, nlags=nlags)
    return autocorr


def plot_auto(data, file):
    fig, axes = plt.subplots(1, 1, figsize=(12, 4))
    for set in data:
        axes.plot(set["auto_data"], color=set["color"], label=set["name"])
    axes.legend()
    axes.set_title("Autocorrelation")
    plt.savefig("{0}.png".format(file))


def autocorrelation(dir, data, data_feature_output):
    """
    args:
    :param dataset: dataset name
    :param data: List of dictionaries (one dictionary per features with keys 'data feature', 'name' and 'color')
    :return: pass
    """
    # create folder to save files
    eval_dir = dir.rsplit("/", 1)[0]
    epoch = dir.rsplit("/", 1)[1]
    dir = "{0}/{1}".format(eval_dir, "autocorrelation")
    if not os.path.exists(dir):
        os.makedirs(dir)
    feature_dim = 0
    for f in data_feature_output:
        if f.type_ == output.OutputType.DISCRETE:
            feature_dim += f.dim
            continue
        data_to_plot = []
        for set in data:
            path = set['data_path']
            nr_samples = len(set['data_gen_flag'])
            data_auto = np.zeros((0, set['data_gen_flag'].shape[1]))
            for idx in range(nr_samples):
                feature = np.load("{}/{}_data_feature.npy".format(path, idx))
                auto = np.expand_dims(autocorr(feature[:, feature_dim]), 0)
                data_auto = np.concatenate((data_auto, auto), axis=0)
            data_avg_auto = np.mean(data_auto, axis=0)
            data_to_plot.append({"auto_data": data_avg_auto, "name": set['name'], "color": set['color']})
        file = "{0}/feature_{1}".format(dir, feature_dim)
        if not os.path.exists(file):
            os.makedirs(file)
        file = "{0}/{1}".format(file, epoch)
        plot_auto(data=data_to_plot, file=file)
        feature_dim += f.dim


######################## SEQUENCE LENGTH ########################


def plot_seq_len(data, file):
    fig, axes = plt.subplots(1, 1, figsize=(12, 4))
    for set in data:
        data_bins = set['data_bins']
        x = np.arange(start=0, stop=len(data_bins), step=1)
        axes.bar(x, data_bins, color=set['color'], label=set['name'], alpha=0.5)
    axes.legend()
    axes.set_title("Sequence Length")
    plt.savefig('{0}.png'.format(file))


def sequence_length(dir, data):
    """
    :param dataset: dataset name
    :param data: List of dictionaries (one dictionary per features with keys 'data_gen_flag', 'name' and 'color')
    :return: pass
    """
    # create folder to save files
    eval_dir = dir.rsplit("/", 1)[0]
    epoch = dir.rsplit("/", 1)[1]
    dir = "{0}/{1}".format(eval_dir, "sequence_length")
    if not os.path.exists(dir):
        os.makedirs(dir)
    data_to_plot = []
    for set in data:
        data_gen_flag = set["data_gen_flag"]
        len = np.count_nonzero(data_gen_flag, axis=1)
        data_bins = np.bincount(len)[:50]
        data_to_plot.append({"data_bins": data_bins, "name": set['name'], "color": set['color']})
    file = "{0}/epoch_{1}".format(dir, epoch)
    plot_seq_len(data_to_plot, file)


######################## CROSS MEASUREMENT ########################


# calculate pearson coefficient
def pearson(x, y):
    return stats.pearsonr(x, y)


# calculate and plot cdf from list
def plot_cdf(data, file):
    fig, axes = plt.subplots(1, 1, figsize=(12, 8))
    for set in data:
        pdf = set['pdf']
        Y = np.cumsum(pdf)
        X = np.arange(pdf.shape[0])
        axes.plot(X, Y, color=set['color'], label=set['name'])
    axes.legend()
    axes.set_title("CDF")
    plt.savefig('{0}.png'.format(file))


def cross_measurement(dir, data, nr_bins, data_feature_output):
    """
    :param dataset: Name of the dataset
    :param data: List of dictionaries (one dictionary per features with keys
    'data_features', data_gen_flag', 'name' and 'color')
    :param nr_bins:
    :return:
    """
    # create folder to save files
    eval_dir = dir.rsplit("/", 1)[0]
    epoch = dir.rsplit("/", 1)[1]
    dir = "{0}/{1}".format(eval_dir, "cross_meas_correlation")
    if not os.path.exists(dir):
        os.makedirs(dir)
    # calculate number of features
    nr_features = 0
    nr_features += [feature.dim for feature in data_feature_output]
    for f_1 in range(nr_features):
        for f_2 in range(f_1 + 1, nr_features):
            data_to_plot = []
            for set in data:
                path = set['data_path']
                nr_samples = len(set['data_gen_flag'])
                data_gen_flag = set['data_gen_flag']
                pearsons = np.zeros(nr_bins)
                for idx in range(nr_samples):
                    feature = np.load("{}/{}_data_feature.npy".format(path, idx))
                    sequence = data_gen_flag[idx, :]
                    length = np.count_nonzero(sequence)
                    x = feature[:length, f_1]
                    y = feature[:length, f_2]
                    if len(x) < 2 or len(y) < 2:
                        continue
                    pear, _ = pearson(x, y)
                    if pear >= 0:
                        bin = int((pear * (nr_bins / 2)) + (nr_bins / 2))
                        if bin > (len(pearsons) - 1):
                            pearsons[bin - 1] += 1
                        else:
                            pearsons[bin] += 1
                    elif math.isnan(pear):
                        pass
                    else:
                        bin = int((pear + 1) * (nr_bins / 2))
                        pearsons[bin] += 1
                pearsons /= nr_samples
                data_to_plot.append({"pdf": pearsons, 'name': set['name'], 'color': set['color']})
            file = "{0}/feature{1}_feature{2}".format(dir, f_1, f_2)
            if not os.path.exists(file):
                os.makedirs(file)
            file = "{0}/{1}".format(file, epoch)
            plot_cdf(data=data_to_plot, file=file)


######################## MEASUREMENT AND METADATA DISTRIBUTION ########################


def plot_distribution(data, file, title, normalization=None):
    fig, axes = plt.subplots(1, 1, figsize=(12, 4))
    bins = None
    for set in data:
        bins = set['bins']
        y_pos = np.arange(len(bins))
        # y_pos = np.linspace(-1, 1, len(bins))
        axes.bar(y_pos, bins, color=set['color'], label=set['name'], alpha=0.5)
    axes.legend()
    axes.set_title(title)
    if normalization is not None:
        x_ticks = range(0, len(bins) + 1, int(len(bins) / 4))
        if normalization == output.Normalization.ZERO_ONE:
            x_labels = ['0', '0.25', '0.5', '0.75', '1']
        else:
            x_labels = ['-1', '-0.5', '0', '0.5', '1']
        plt.xticks(x_ticks, x_labels)
    plt.savefig('{0}.png'.format(file))


def measurement_distribution(dir, data, feature_output, nr_bins=100):
    # create folder to save files
    eval_dir = dir.rsplit("/", 1)[0]
    epoch = dir.rsplit("/", 1)[1]
    dir = "{0}/{1}".format(eval_dir, "measurement_distribution")
    if not os.path.exists(dir):
        os.makedirs(dir)
    dim = 0
    counter = 0
    for f in feature_output:
        if f.type_ == output.OutputType.DISCRETE:
            dim += f.dim
            continue
        data_to_plot = []
        for set in data:
            path = set['data_path']
            nr_samples = len(set['data_gen_flag'])
            bins = np.zeros(nr_bins)
            for idx in range(len(nr_samples)):
                feature = np.load("{}/{}_data_feature.npy".format(path, idx))
                max = feature.max()
                min = feature.min()
                value = (max + min) / 2
                if f.normalization == output.Normalization.ZERO_ONE:
                    bin = int(value * nr_bins)
                    bins[bin] += 1
                else:
                    if value >= 0:
                        bin = int((value * (nr_bins / 2)) + (nr_bins / 2))
                    else:
                        bin = int((value + 1) * (nr_bins / 2))
                    bins[bin] += 1
            data_to_plot.append({'bins': bins, 'name': set['name'], 'color': set['color']})
        dim += f.dim
        file = "{0}/feature_{1}".format(dir, counter)
        if not os.path.exists(file):
            os.makedirs(file)
        file = "{0}/{1}".format(file, epoch)
        plot_distribution(data=data_to_plot, file=file,
                          title="measurement_distribution", normalization=f.normalization)
        counter += 1


def metadata_distribution(dir, data, attribute_output):
    """
    :param data: List of dictionaries (one dictionary per features with keys 'data_attribute', 'name' and 'color')
    :param attribute_output: description of attributes
    :param dataset: name of the dataset
    :return: pass
    """
    # create folder to save files
    eval_dir = dir.rsplit("/", 1)[0]
    epoch = dir.rsplit("/", 1)[1]
    dir = "{0}/{1}".format(eval_dir, "metadata_distribution")
    if not os.path.exists(dir):
        os.makedirs(dir)
    # dir = "{0}/{1}".format(dir, epoch)
    dim = 0
    counter = 0
    for i in attribute_output:
        dim_range = dim + i.dim
        data_to_plot = []
        for set in data:
            path = set['data_path']
            nr_samples = len(set['data_gen_flag'])
            bins = np.zeros(dim_range)
            for idx in range(nr_samples):
                attribute = np.load("{}/{}_data_attribute.npy".format(path, idx))
                attribute = attribute[dim:dim_range]
                index = np.argmax(attribute)
                bins[index] += 1
            data_to_plot.append({'bins': bins, 'name': set['name'], 'color': set['color']})
        file = "{0}/attribute_{1}".format(dir, counter)
        if not os.path.exists(file):
            os.makedirs(file)
        file = "{0}/{1}".format(file, epoch)
        plot_distribution(data=data_to_plot, file=file,
                          title="metadata_distribution")
        dim += i.dim
        counter += 1


######################## NEAREST NEIGHBOR ########################

def plot_nearest_neighbors(data, file, real_path, sample_path, feature_dim):
    fig, axs = plt.subplots(3, 4, figsize=(12, 4))
    plt.tight_layout()
    axs[0, 0].set_title('Gen. Sample')
    axs[0, 1].set_title('1st NN')
    axs[0, 2].set_title('2nd NN')
    axs[0, 3].set_title('3rd NN')
    row_counter = 0
    for nn in data:
        sample = np.load("{}/{}_data_feature.npy".format(sample_path, nn['sample_idx']))
        sample = sample[:, feature_dim]
        first_n = np.load("{}/{}_data_feature.npy".format(real_path, nn['nn_indices'][0]))
        first_n = first_n[:, feature_dim]
        second_n = np.load("{}/{}_data_feature.npy".format(real_path, nn['nn_indices'][1]))
        second_n = second_n[:, feature_dim]
        third_n = np.load("{}/{}_data_feature.npy".format(real_path, nn['nn_indices'][2]))
        third_n = third_n[:, feature_dim]
        axs[row_counter, 0].plot(sample)
        axs[row_counter, 1].plot(first_n)
        axs[row_counter, 2].plot(second_n)
        axs[row_counter, 3].plot(third_n)
        row_counter += 1
    plt.savefig("{0}.png".format(file))


def nearest_neighbors(dir, data, data_feature_outputs):
    # create folder to save files
    eval_dir = dir.rsplit("/", 1)[0]
    epoch = dir.rsplit("/", 1)[1]
    dir = "{0}/{1}".format(eval_dir, "nearest_neighbor")
    if not os.path.exists(dir):
        os.makedirs(dir)
    feature_dim = 0
    for f in range(len(data_feature_outputs)):
        if data_feature_outputs[f].type_ == output.OutputType.DISCRETE:
            feature_dim += data_feature_outputs[f].dim
            continue
        data_to_plot = []
        for i in range(3):
            real_path = data[0]['data_path']
            sample_path = data[1]['data_path']
            nr_samples = len(data[0]['data_gen_flag'])
            # select random sample
            sample_idx = random.randint(0, nr_samples - 1)
            sample_feature = np.load("{}/{}_data_feature.npy".format(sample_path, sample_idx))
            sample_feature = sample_feature[:, feature_dim]
            # calculate distance to all real samples
            dist = np.zeros(nr_samples)
            for idx in range(nr_samples):
                real_feature = np.load("{}/{}_data_feature.npy".format(real_path, idx))
                real_feature = real_feature[:, feature_dim]
                mse = metrics.mean_squared_error(real_feature, sample_feature)
                dist[idx] = mse
            # get indices with lowest mse
            dist_ind = np.argsort(dist)
            data_to_plot.append({"sample_index": sample_idx, "nn_indices": dist_ind})
        file = "{0}/feature_{1}".format(dir, f)
        if not os.path.exists(file):
            os.makedirs(file)
        file = "{0}/{1}".format(file, epoch)
        plot_nearest_neighbors(data=data_to_plot, file=file,
                               real_data_features=real_data_features, sampled_data_features=sampled_data_features,
                               feature_dim=f)


######################## LOAD DATA AND SAMPLES ########################

dataset_name = 'transactions'
gan_type = 'RNN'
normalize = False
epoch_id = 400

data_path = "data/{}".format(dataset_name)
data_gen_flag = np.load("{}/data_gen_flag.npy".format(data_path))
with open("{}/data_feature_output.pkl".format(data_path), "rb") as f:
    data_feature_outputs = pickle.load(f)
if normalize:
    data_path = "{}/normalized".format(data_path)

sample_path = 'runs/{}/{}/1/checkpoint/epoch_{}'.format(dataset_name, gan_type, epoch_id)
sampled_gen_flags = np.load("{}/sampled_gen_flags.npy".format(sample_path))

# generate list of dictionaries
data = []
# append real data
data.append({
    'data_path': data_path,
    'data_gen_flag': data_gen_flag,
    'color': 'yellow',
    'name': 'REAL'
})
# append sampled data
data.append({
    'data_path': sample_path,
    'data_gen_flag': sampled_gen_flags,
    'color': 'blue',
    'name': 'torchDoppelGANger'
})
# create folder to save data
run_dir = "{}/{}".format(sample_path.split("/")[2], sample_path.split("/")[3])
epoch_id = sample_path.split("/")[5]
evaluation_dir = "evaluation/{0}/{1}".format(dataset_name, run_dir)
if not os.path.exists(evaluation_dir):
    os.makedirs(evaluation_dir)
evaluation_dir = "{0}/{1}".format(evaluation_dir, epoch_id)

# call methods
#autocorrelation(dir=evaluation_dir, data=data, data_feature_output=data_feature_outputs)
sequence_length(evaluation_dir, data)