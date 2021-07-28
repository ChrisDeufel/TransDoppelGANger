import math
import random
from scipy import stats
from sklearn import metrics
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import statsmodels.api as sm
import output
from load_data import load_data
from util import add_gen_flag, normalize_per_sample
sys.modules["output"] = output

### AUTOCORRELATION ###
def autocorr(x, nlags):
    autocorr = sm.tsa.acf(x, nlags=nlags)
    return autocorr

def plot_auto(data, file):
    fig, axes = plt.subplots(1, 1, figsize=(12, 4))
    for set in data:
        axes.plot(set["auto_data"], color=set["color"], label=set["name"])
    axes.legend()
    axes.set_title("Autocorrelation")
    plt.savefig("{0}.png".format(file))

def autocorrelation(dataset, data, shape):
    """
    args:
    :param dataset: dataset name
    :param data: List of dictionaries (one dictionary per features with keys 'data feature', 'name' and 'color')
    :return: pass
    """
    for f in range(shape[2]):
        data_to_plot = []
        for set in data:
            data_feature = set['data_feature']
            data_avg_auto = np.zeros(data_feature.shape[1])
            for i in range(data_feature.shape[0]):
                features = np.reshape(data_feature[i, :, f], (data_feature.shape[1]))
                auto = autocorr(x=features, nlags=data_feature.shape[1])
                if np.isnan(auto).any():
                    continue
                data_avg_auto += auto

            data_avg_auto /= data_feature.shape[0]
            data_to_plot.append({"auto_data": data_avg_auto, "name": set['name'], "color": set['color']})
        file = "autocorrelation_{0}_feature_{1}".format(dataset, f)
        plot_auto(data=data_to_plot, file=file)

### SEQUENCE LENGTH ###
def plot_seq_len(data, file):
    fig, axes = plt.subplots(1, 1, figsize=(12, 4))
    for set in data:
        data_bins = set['data_bins']
        x = np.arange(start=0, stop=len(data_bins), step=1)
        axes.bar(x, data_bins, color=set['color'], label=set['name'])
    axes.legend()
    axes.set_title("Sequence Length")
    plt.savefig('{0}_Sequence_Length.png'.format(file))

def sequence_length(dataset, data):
    """
    :param dataset: dataset name
    :param data: List of dictionaries (one dictionary per features with keys 'data_gen_flag', 'name' and 'color')
    :return: pass
    """
    data_to_plot = []
    for set in data:
        data_gen_flag = set["data_gen_flag"]
        len = np.count_nonzero(data_gen_flag, axis=1)
        data_bins = np.bincount(len)[:50]
        data_to_plot.append({"data_bins": data_bins, "name": set['name'], "color": set['color']})
    plot_seq_len(data_to_plot, dataset)

### PEARSON CORRELATION ###
# calculate pearson coefficient
def pearson(x, y):
    return stats.pearsonr(x, y)

# calculate and plot cdf from list
def plot_cdf(data, file):
    fig, axes = plt.subplots(1, 1, figsize=(12, 4))
    for set in data:
        pdf = set['pdf']
        Y = np.cumsum(pdf)
        X = np.arange(pdf.shape[0])
        axes.plot(X, Y, color=set['color'], label=set['name'])
    axes.legend()
    axes.set_title("CDF")
    plt.savefig('{0}.png'.format(file))

def cross_measurement(dataset, data, nr_bins):
    """
    :param dataset: Name of the dataset
    :param data: List of dictionaries (one dictionary per features with keys
    'data_features', data_gen_flag', 'name' and 'color')
    :param nr_bins:
    :return:
    """
    data_to_plot = []
    for set in data:
        data_feature = set['data_feature']
        data_gen_flag = set['data_gen_flag']
        pearsons = np.zeros(nr_bins)
        for i in range(data_feature.shape[0]):
            sequence = data_gen_flag[i, :]
            length = np.count_nonzero(sequence)
            # the selection of features here is still manual
            # TODO: automate selection
            x = data_feature[i, :length, 0]
            y = data_feature[i, :length, 3]
            if len(x) < 2:
                continue
            pear, _ = pearson(x, y)
            if pear >= 0:
                bin = int((pear*(nr_bins/2))+(nr_bins/2))
                if bin > (len(pearsons)-1):
                    pearsons[bin-1] += 1
                else:
                    pearsons[bin] += 1
            elif math.isnan(pear):
                pass
            else:
                bin = int((pear+1) * (nr_bins / 2))
                pearsons[bin] += 1
        pearsons /= data_feature.shape[0]
        data_to_plot.append({"pdf": pearsons, 'name': set['name'], 'color': set['color']})
    plot_cdf(data=data_to_plot, file=dataset)


### PLOT MEASUREMENT DISTRIBUTION OR METADATA DISTRIBUTION ###
def plot_distribution(data, file):
    fig, axes = plt.subplots(1, 1, figsize=(12, 4))
    for set in data:
        bins = set['bins']
        y_pos = np.arange(len(bins))
        axes.bar(y_pos, bins, color=set['color'], label=set['name'], alpha=0.5)
    axes.legend()
    axes.set_title("METADATA DISTRIBUTION")
    plt.savefig('metadata_distribution_{0}.png'.format(file))

def metadata_distribution(data, attribute_output, dataset):
    """
    :param data: List of dictionaries (one dictionary per features with keys 'data_attribute', 'name' and 'color')
    :param attribute_output: description of attributes
    :param dataset: name of the dataset
    :return: pass
    """
    dim = 0
    counter = 0
    for i in attribute_output:
        dim_range = dim + i.dim
        data_to_plot = []
        for set in data:
            data_attribute = set['data_attribute']
            bins = np.zeros(i.dim)
            for j in range(data_attribute.shape[0]):
                bins += data_attribute[j, dim:dim_range]
            data_to_plot.append({'bins': bins, 'name': set['name'], 'color': set['color']})
        plot_distribution(data_to_plot, "{0}_attribute_{1}".format(dataset, counter))
        dim += i.dim
        counter += 1


### MEASUREMENT METADATA CORRELATIONS ###
def wasserstein_distance(x, y):
    return stats.wasserstein_distance(x, y)

def meta_meas_corr(data, data_attribute_outputs, data_feature_outputs, dataset):
    """
    :param data: List of dictionaries (one dictionary per features with keys 'data_attribute',
    'data_features', 'name' and 'color')
    :param nr_bins:
    :param data_attribute_outputs: Description of attributes
    :param data_feature_outputs: Descriptions of features
    :param dataset: Name of dataset
    :return: pass
    """

    attribute_counter = 0
    for a in range(len(data_attribute_outputs)):
        if data_attribute_outputs[a].type_ == output.OutputType.CONTINUOUS:
            attribute_counter += 1
            continue
        else:
            feature_counter = 0
            for f in range(len(data_feature_outputs)):
                if data_feature_outputs[f].type_ == output.OutputType.DISCRETE:
                    feature_counter += data_feature_outputs[f].dim
                    continue
                for i in range(data_attribute_outputs[a].dim):
                    data_to_plot = []
                    for set in data:
                        data_attribute = set['data_attribute']
                        data_feature = set['data_feature']
                        sums = []
                        if np.sum(data_attribute[:, attribute_counter+i]) < 1:
                            continue
                        sample_counter = 0
                        for j in range(len(data_feature)):
                            if data_attribute[j][attribute_counter+i] == 1:
                                sums.append(np.sum(data_feature[j, :, feature_counter]))
                                sample_counter += 1
                        sums = np.array(sums).astype(np.int)
                        pdf = np.bincount(sums).astype(np.float) / sample_counter
                        data_to_plot.append({'pdf': pdf, 'name': set['name'], 'color': set['color']})
                    plot_cdf(data_to_plot,'meta_meas_corr_{0}_attribute{1}_dim{2}_feature{3}'.format(dataset, a, i, f))
                feature_counter += data_feature_outputs[f].dim

### DG DOES NOT OVERFIT ###
# TODO: find out squared error of what exactly
# always calculate and display top 3 nearest neighbors for 3 samples
# does not work for discrete features
def plot_nearest_neighbors(data, file, real_data_features, sampled_data_features, feature_dim):
    fig, axs = plt.subplots(3, 4, figsize=(12, 4))
    plt.tight_layout()
    axs[0, 0].set_title('Gen. Sample')
    axs[0, 1].set_title('1st NN')
    axs[0, 2].set_title('2nd NN')
    axs[0, 3].set_title('3rd NN')
    row_counter = 0
    for nn in data:
        sample = sampled_data_features[nn['sample_index'], :, feature_dim]
        index = nn['nn_indices'][0]
        first_n = real_data_features[index, :, feature_dim]
        second_n = real_data_features[nn['nn_indices'][1], :, feature_dim]
        third_n = real_data_features[nn['nn_indices'][2], :, feature_dim]
        axs[row_counter, 0].plot(sample)
        axs[row_counter, 1].plot(first_n)
        axs[row_counter, 2].plot(second_n)
        axs[row_counter, 3].plot(third_n)
        row_counter += 1
    plt.savefig("nn_{0}.png".format(file))


def nearest_neighbors(real_data_features, sampled_data_features, data_feature_outputs, dataset):
    feature_dim = 0
    for f in range(len(data_feature_outputs)):
        if data_feature_outputs[f].type_ == output.OutputType.DISCRETE:
            feature_dim += data_feature_outputs[f].dim
            continue
        data_to_plot = []
        for i in range(3):
            # select random sample
            sample_nr = random.randint(0, sampled_data_features.shape[0])
            sample_feature = sampled_data_features[sample_nr, :, feature_dim]
            # calculate distance to all real samples
            dist = np.zeros(real_data_features.shape[0])
            for s in range(real_data_features.shape[0]):
                real_feature = real_data_features[s, :, feature_dim]
                #mse = ((sample_feature - real_feature)**2).mean()
                mse = metrics.mean_squared_error(real_feature, sample_feature)
                dist[s] = mse
            # get indices with lowest mse
            dist_ind = np.argsort(dist)
            data_to_plot.append({"sample_index": sample_nr, "nn_indices": dist_ind})
        plot_nearest_neighbors(data=data_to_plot, file="{0}_feature{1}".format(dataset, f),
                               real_data_features=real_data_features, sampled_data_features=sampled_data_features,
                               feature_dim=f)

### RESOURCE COSTS ###
def mean_square_error(x, y):
    return metrics.mean_squared_error(x, y)

#TODO: apparently average mse ?!
def mse_autocorrelation(real_data_features, sample_data_features, data_feature_outputs, file):
    feature_dim = 0
    mses = []
    columns = []
    for f in range(len(data_feature_outputs)):
        if data_feature_outputs[f].type_ == output.OutputType.DISCRETE:
            feature_dim += data_feature_outputs[f].dim
            continue
        real_feature = real_data_features[:, :, f]
        sample_feature = sample_data_features[:, :, f]
        mse = metrics.mean_squared_error(real_feature, sample_feature)
        mses.append(mse)
        columns.append('feature_{0}'.format(f))
    mses = np.asarray(mses)/real_data_features.shape[0]
    mses = np.reshape(mses, (1, len(mses)))
    columns = np.asarray(columns)
    df = pd.DataFrame(data=mses, columns=columns)
    df.to_csv('mse_auto_{0}.csv'.format(file), sep=';')


# load web dataset for testing
dataset = 'web'

# load original data
(data_feature, data_attribute, data_gen_flag, data_feature_outputs, data_attribute_outputs) = \
        load_data("data/{0}".format(dataset))

# if normalization needed
"""
(data_feature, data_attribute, data_attribute_outputs, real_attribute_mask) = \
        normalize_per_sample(data_feature, data_attribute, data_feature_outputs, data_attribute_outputs)
"""
# load generated data
sample_path = 'runs/web_1/checkpoint/epoch_395/generated_samples.npz'
sampled_data = np.load(sample_path)

sampled_features = sampled_data['sampled_features']
sampled_attributes = sampled_data['sampled_attributes']
sampled_gen_flags = sampled_data['sampled_gen_flags']
sampled_lengths = sampled_data['sampled_lengths']

# generate list of dictionaries
data = []
# append real data
data.append({
    'data_feature': data_feature,
    'data_attribute': data_attribute,
    'color': 'blue',
    'name': 'REAL'
})
# append sampled data
data.append({
    'data_feature': sampled_features,
    'data_attribute': sampled_attributes,
    'color': 'yellow',
    'name': 'DoppelGANger'
})
autocorrelation(dataset=dataset, data=data, shape=data_feature.shape)
#metadata_distribution(data=data, attribute_output=data_attribute_outputs, dataset=dataset)
#meta_meas_corr(data=data, nr_bins=0, data_attribute_outputs=data_attribute_outputs,
#               data_feature_outputs=data_feature_outputs, dataset=dataset)
#nearest_neighbors(real_data_features=data_feature, sampled_data_features=sampled_features,
#                  data_feature_outputs=data_feature_outputs, dataset=dataset)
#mse_autocorrelation(real_data_features=data_feature, sample_data_features=sampled_features,
#                    data_feature_outputs=data_feature_outputs, file=dataset)

