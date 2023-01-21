import math
import statistics
import random
from scipy import stats
from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import statsmodels.api as sm
import output
from load_data import load_data
from util import add_gen_flag, normalize_per_sample
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

sys.modules["output"] = output

np.seterr(divide='ignore', invalid='ignore')


### QQ Plot ###
def plot_qq_plot(data, file, metric):
    fig, axes = plt.subplots(1, 1, figsize=(12, 4))
    m_max = np.max((data[0]["feature_metric"], data[1]["feature_metric"]))
    m_min = np.min((data[0]["feature_metric"], data[1]["feature_metric"]))
    # axes.scatter(np.sort(data[0]["feature_metric"]), np.sort(data[1]["feature_metric"]))
    axes.plot([0, 1], [0, 1], transform=axes.transAxes, color="red", alpha=0.5)
    axes.scatter(np.sort(data[0]["feature_metric"]), np.sort(data[1]["feature_metric"]))
    plt.xlim(m_min, m_max)
    plt.ylim(m_min, m_max)
    plt.xlabel('Real')
    plt.ylabel('Generated')
    axes.set_title(metric)
    plt.savefig("{0}.png".format(file))


def qq_plot(dir, data, data_feature_output, metric='mean'):
    # create folder to save files
    feature_dim = 0
    for f in data_feature_output:
        if f.type_ == output.OutputType.DISCRETE:
            feature_dim += f.dim
            continue
        data_to_plot = []
        for set in data:
            data_feature = set['data_feature']
            data_feature = data_feature[:, :, feature_dim]
            if metric == 'mean':
                f_metric = np.mean(data_feature, axis=1)
            elif metric == 'variance':
                f_metric = np.var(data_feature, axis=1)
            elif metric == 'skewness':
                f_metric = stats.skew(data_feature, axis=1)
            else:
                f_metric = stats.kurtosis(data_feature, axis=1)
            data_to_plot.append({"feature_metric": f_metric, "name": set['name'], "color": set['color']})
        file = "{}/QQ_{}_feature_{}".format(dir, metric, feature_dim)
        plot_qq_plot(data=data_to_plot, file=file, metric=metric)


### TSNE/ PCA Embedding ###
def plot_embedding(data, file, embedding):
    fig, axes = plt.subplots(1, 1, figsize=(12, 4))
    counter = 0
    for set in reversed(data):
        if counter == 0:
            alpha = 1
        else:
            alpha = 0.3
        axes.scatter(set["embedded_features"][:, 0], set["embedded_features"][:, 1], color=set["color"],
                     label=set["name"], alpha=alpha)
        counter += 1
    axes.legend()
    axes.set_title(embedding)
    plt.savefig("{0}.png".format(file))


def embedding(dir, data, data_feature_output, embedding='TSNE'):
    feature_dim = 0
    for f in data_feature_output:
        if f.type_ == output.OutputType.DISCRETE:
            feature_dim += f.dim
            continue
        data_to_plot = []
        for set in data:
            data_feature = set['data_feature']
            data_feature = data_feature[:, :, feature_dim]
            if embedding == 'TSNE':
                f_embedded = TSNE(n_components=2, init='random').fit_transform(data_feature)
            else:
                f_embedded = PCA(n_components=2).fit_transform(data_feature)
            data_to_plot.append({"embedded_features": f_embedded, "name": set['name'], "color": set['color']})
        file = "{}/{}_feature_{}".format(dir, embedding, feature_dim)
        plot_embedding(data=data_to_plot, file=file, embedding=embedding)


### WASSERSTEIN DISTANCE ###
def emd(dir, data, data_feature_output, epoch):
    """
    args:
    :param dataset: dataset name
    :param data: List of dictionaries (one dictionary per features with keys 'data feature', 'name' and 'color')
    :return: pass
    """
    feature_dim = 0
    for f in data_feature_output:
        if f.type_ == output.OutputType.DISCRETE:
            feature_dim += f.dim
            continue
        data_to_plot = []
        for set in data:
            data_feature = set['data_feature']
            data_feature = data_feature[:, :, feature_dim]
            feature_mean = np.mean(data_feature, axis=0)
            data_to_plot.append({"feature_mean": feature_mean, "name": set['name'], "color": set['color']})
        emd = stats.wasserstein_distance(data_to_plot[0]['feature_mean'], data_to_plot[1]['feature_mean'])
        file = "{0}/emds_feature_{1}".format(dir, feature_dim)
        writer_file = open("{}.txt".format(file), "a")
        writer_file.write("{}: {}\n".format(epoch, emd))
        writer_file.close()
        feature_dim += f.dim


### AUTOCORRELATION ###
def part_autocorr(x, nlags):
    return sm.tsa.pacf(x, nlags=nlags)


def autocorr(x, nlags):
    return sm.tsa.acf(x, nlags=nlags)


def plot_auto(data, file, partial):
    fig, axes = plt.subplots(1, 1, figsize=(12, 4))
    for set in data:
        axes.plot(np.arange(1, len(set["auto_data"])), set["auto_data"][1:], color=set["color"], label=set["name"])
    axes.legend()
    if partial:
        axes.set_title("Partial-Autocorrelation")
    else:
        axes.set_title("Autocorrelation")
    # plt.xlim(1, len(data[0]["auto_data"]-2))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)
    plt.savefig("{}_nlags_{}.png".format(file, len(data[0]["auto_data"]) - 1))


def autocorrelation(dir, data, data_feature_output, n_lags, partial=False):
    """
    args:
    :param dataset: dataset name
    :param data: List of dictionaries (one dictionary per features with keys 'data feature', 'name' and 'color')
    :return: pass
    """
    feature_dim = 0
    for f in data_feature_output:
        if f.type_ == output.OutputType.DISCRETE:
            feature_dim += f.dim
            continue
        data_to_plot = []
        for set in data:
            data_feature = set['data_feature']
            data_gen_flag = set['data_gen_flag']
            data_feature = data_feature[:, :, feature_dim]
            auto = 0
            counter = 0
            for i in range(len(data_feature)):
                len_seq = np.count_nonzero(data_gen_flag[i, :])
                if partial:
                    if (len_seq//2) <= n_lags:
                        continue
                else:
                    if len_seq < n_lags:
                        continue
                if partial:
                    try:
                        auto += part_autocorr(data_feature[i, :len_seq + 1], n_lags)
                        counter += 1
                    except np.linalg.LinAlgError:
                        continue

                else:

                    auto_sample = autocorr(data_feature[i, :len_seq+1], n_lags)
                    if np.isnan(auto_sample).any():
                        print(counter)
                        continue
                    auto += auto_sample
                    counter += 1
            if counter != 0:
                auto /= counter
            else:
                auto = np.zeros(n_lags)
            # autos = autos[~np.isnan(autos).any(axis=1), :]
            # data_avg_auto = np.mean(autos, axis=0)
            data_to_plot.append({"auto_data": auto, "name": set['name'], "color": set['color']})
        if partial:
            file = "{0}/partial_auto_feature_{1}".format(dir, feature_dim)
        else:
            file = "{0}/auto_feature_{1}".format(dir, feature_dim)
        plot_auto(data=data_to_plot, file=file, partial=partial)
        feature_dim += f.dim


### SEQUENCE LENGTH ###
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
    data_to_plot = []
    for set in data:
        data_gen_flag = set["data_gen_flag"]
        len = np.count_nonzero(data_gen_flag, axis=1)
        data_bins = np.bincount(len)
        data_to_plot.append({"data_bins": data_bins, "name": set['name'], "color": set['color']})
    file = "{0}/sequence_length".format(dir)
    plot_seq_len(data_to_plot, file)


### PEARSON CORRELATION ###
# calculate pearson coefficient
def pearson(x, y):
    return stats.pearsonr(x, y)


# calculate and plot cdf from list
def plot_cdf(data, file, normalization):
    fig, axes = plt.subplots(1, 1, figsize=(12, 8))
    pdf = None
    for set in data:
        pdf = set['pdf']
        Y = np.cumsum(pdf)
        if normalization == output.Normalization.MINUSONE_ONE:
            X = np.linspace(-1, 1, pdf.shape[0])
        else:
            X = np.linspace(0, 1, pdf.shape[0])
        axes.plot(X, Y, color=set['color'], label=set['name'])

    if normalization is not None:
        x_ticks = range(0, len(pdf) + 1, int(len(pdf) / 4))
        if normalization == output.Normalization.ZERO_ONE:
            x_labels = ['0', '0.25', '0.5', '0.75', '1']
        else:
            x_labels = ['-1', '-0.5', '0', '0.5', '1']
        plt.xticks(x_ticks, x_labels)
    axes.legend()
    axes.set_title("CDF")
    plt.savefig('{0}.png'.format(file))


# TODO: beautify method
def cross_measurement(dir, data, nr_bins):
    """
    :param dataset: Name of the dataset
    :param data: List of dictionaries (one dictionary per features with keys
    'data_features', data_gen_flag', 'name' and 'color')
    :param nr_bins:
    :return:
    """
    nr_features = data[0]['data_feature'].shape[2]
    for f_1 in range(nr_features):
        for f_2 in range(f_1 + 1, nr_features):
            data_to_plot = []
            for set in data:
                data_feature = set['data_feature']
                data_gen_flag = set['data_gen_flag']
                pearsons = np.zeros(nr_bins)
                for i in range(data_feature.shape[0]):
                    sequence = data_gen_flag[i, :]
                    length = np.count_nonzero(sequence)

                    x = data_feature[i, :length, f_1]
                    y = data_feature[i, :length, f_2]
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
                # pearsons /= data_feature.shape[0]
                data_to_plot.append({"pdf": pearsons, 'name': set['name'], 'color': set['color']})
            file = "{0}/cross_meas_feature{1}_feature{2}".format(dir, f_1, f_2)
            plot_cdf(data=data_to_plot, file=file)


### PLOT MEASUREMENT DISTRIBUTION OR METADATA DISTRIBUTION ###
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

    else:
        x_ticks = np.arange(0, len(bins), 1)
        x_labels = np.arange(1, len(bins)+1, 1)
    plt.xticks(x_ticks, x_labels)
    plt.savefig('{0}.png'.format(file))


def plot_distribution_2(data, file, title, normalization=None):
    fig, axes = plt.subplots(len(data) - 1, 1, figsize=(12, 4))
    bins = None
    # real data must always the first dic in the data list
    for i in range(len(axes)):
        ax = axes[i]
        # plot real data
        set = data[0]
        bins = set['bins']
        y_pos = np.arange(len(bins))
        ax.bar(y_pos, bins, color=set['color'], label=set['name'], alpha=0.5)
        # plot current fake data
        set = data[i + 1]
        bins = set['bins']
        y_pos = np.arange(len(bins))
        ax.bar(y_pos, bins, color=set['color'], label=set['name'], alpha=0.5)
        ax.legend()
    axes.set_title(title)

    if normalization is not None:
        x_ticks = range(0, len(bins) + 1, int(len(bins) / 4))
        if normalization == output.Normalization.ZERO_ONE:
            x_labels = ['0', '0.25', '0.5', '0.75', '1']
        else:
            x_labels = ['-1', '-0.5', '0', '0.5', '1']
        plt.xticks(x_ticks, x_labels)
    plt.savefig('{0}.png'.format(file))


# def measurement_distribution(dir, data, feature_output, nr_bins=100):
#     dim = 0
#     counter = 0
#     for f in feature_output:
#         if f.type_ == output.OutputType.DISCRETE:
#             dim += f.dim
#             continue
#         data_to_plot = []
#         for set in data:
#             data_feature = set['data_feature']
#             data_gen_flag = set['data_gen_flag']
#             bins = np.zeros(nr_bins)
#             for j in range(len(data_feature)):
#                 feature = data_feature[j, :np.count_nonzero(data_gen_flag[j, :]), dim]
#                 # max = feature.max()
#                 # min = feature.min()
#                 # value = (max + min) / 2
#                 for value in feature:
#                     if f.normalization == output.Normalization.ZERO_ONE:
#                         bin = int(value * nr_bins)
#                         if bin < 0:
#                             bin = 0
#                         if bin >= nr_bins:
#                             bins[nr_bins - 1] += 1
#                         else:
#                             bins[bin] += 1
#                     else:
#                         if value >= 0:
#                             bin = int((value * (nr_bins / 2)) + (nr_bins / 2))
#                         else:
#                             bin = int((value + 1) * (nr_bins / 2))
#                         if bin >= nr_bins:
#                             bins[nr_bins - 1] += 1
#                         elif bin < 0:
#                             bins[0] += 1
#                         else:
#                             bins[bin] += 1
#             data_to_plot.append({'bins': bins, 'name': set['name'], 'color': set['color']})
#         dim += f.dim
#         file = "{0}/meas_dis_feature_{1}".format(dir, counter)
#         plot_distribution(data=data_to_plot, file=file,
#                           title="measurement_distribution", normalization=f.normalization)
#         counter += 1


def measurement_distribution(dir, data, feature_output, nr_bins=100):
    dim = 0
    counter = 0
    for f in feature_output:
        if f.type_ == output.OutputType.DISCRETE:
            dim += f.dim
            continue
        data_to_plot = []
        for set in data:
            data_feature = set['data_feature']
            data_gen_flag = set['data_gen_flag']
            bins = np.zeros(nr_bins)
            for j in range(len(data_feature)):
                feature = data_feature[j, :, dim]
                max = feature.max()
                min = feature.min()
                value = (max + min) / 2
                # value = np.mean(feature)
                # for value in feature:
                if f.normalization == output.Normalization.ZERO_ONE:
                    bin = int(value * nr_bins)
                    bins[bin] +=1
                else:
                    if value >= 0:
                        bin = int((value * (nr_bins / 2)) + (nr_bins / 2))
                    else:
                        bin = int((value + 1) * (nr_bins / 2))
                    bins[bin] += 1
            data_to_plot.append({'bins': bins, 'name': set['name'], 'color': set['color']})
        dim += f.dim
        file = "{0}/meas_dis_feature_{1}".format(dir, counter)
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
    dim = 0
    counter = 0
    for i in attribute_output:
        dim_range = dim + i.dim
        data_to_plot = []
        for set in data:
            data_attribute = set['data_attribute']
            data_attribute = data_attribute[:, dim:dim_range]
            indices = np.argmax(data_attribute, axis=1)
            one_hot = np.zeros(data_attribute.shape)
            one_hot[np.arange(indices.size), indices] = 1
            data_attribute = one_hot
            bins = np.sum(data_attribute, axis=0)
            data_to_plot.append({'bins': bins, 'name': set['name'], 'color': set['color']})
        file = "{0}/meta_dis_attribute_{1}".format(dir, counter)
        plot_distribution(data=data_to_plot, file=file,
                          title="metadata_distribution")
        dim += i.dim
        counter += 1


### MEASUREMENT METADATA CORRELATIONS ###
def wasserstein_distance(x, y):
    return stats.wasserstein_distance(x, y)


def meta_meas_corr(dir, data, data_attribute_outputs, data_feature_outputs, plot=True, w_distance=True):
    """
    :param data: List of dictionaries (one dictionary per features with keys 'data_attribute',
    'data_features', 'name' and 'color')
    :param nr_bins:
    :param data_attribute_outputs: Description of attributes
    :param data_feature_outputs: Descriptions of features
    :param dataset: Name of dataset
    :return: pass
    """
    # create folder to save files
    eval_dir = dir.rsplit("/", 1)[0]
    epoch = dir.rsplit("/", 1)[1]
    dir = "{0}/{1}".format(eval_dir, "meta_meas_correlation")
    if not os.path.exists(dir):
        os.makedirs(dir)
    d = []
    columns = []
    attribute_dim = 0
    # loop over every attribute
    for a in range(len(data_attribute_outputs)):
        # skip if attribute is continuous
        attribute = data_attribute_outputs[a]
        if attribute.type_ == output.OutputType.CONTINUOUS:
            attribute_dim += attribute.dim
            continue
        feature_dim = 0
        # loop over all features
        for f in range(len(data_feature_outputs)):
            feature = data_feature_outputs[f]
            # skip if feature is discrete
            if feature.type_ == output.OutputType.DISCRETE:
                feature_dim += feature.dim
                continue
            # real data
            data_attribute_real = data[0]['data_attribute']
            data_feature_real = data[0]['data_feature']
            relevant_attributes_real = data_attribute_real[:, attribute_dim: attribute_dim + attribute.dim]
            # make one hot encoded
            indices = np.argmax(relevant_attributes_real, axis=1)
            one_hot = np.zeros(relevant_attributes_real.shape)
            one_hot[np.arange(indices.size), indices] = 1
            relevant_attributes_real = one_hot
            relevant_features_real = data_feature_real[:, :, feature_dim]
            a_f_real = np.concatenate((relevant_attributes_real, relevant_features_real), axis=1)

            # generated data
            data_attribute_fake = data[1]['data_attribute']
            data_feature_fake = data[1]['data_feature']
            relevant_attributes_fake = data_attribute_fake[:, attribute_dim: attribute_dim + attribute.dim]
            # make one hot encoded
            indices = np.argmax(relevant_attributes_fake, axis=1)
            one_hot = np.zeros(relevant_attributes_fake.shape)
            one_hot[np.arange(indices.size), indices] = 1
            relevant_attributes_fake = one_hot
            relevant_features_fake = data_feature_fake[:, :, feature_dim]
            a_f_fake = np.concatenate((relevant_attributes_fake, relevant_features_fake), axis=1)
            for i in range(attribute.dim):
                # get only features for certain category
                features_real = a_f_real[a_f_real[:, i] == 1, attribute.dim:]
                sum_real = np.sum(features_real, axis=1)
                features_fake = a_f_fake[a_f_fake[:, i] == 1, attribute.dim:]
                sum_fake = np.sum(features_fake, axis=1)
                if sum_real.size != 0 and sum_fake.size != 0:
                    # nr_bins = int(max([np.max(sum_real), np.max(sum_fake)])) + 1
                    minimum = int(min([np.min(sum_real), np.min(sum_fake)]))
                    if minimum < 0:
                        nr_bins = int(max([np.max(sum_real), np.max(sum_fake)]) - minimum) + 1
                    else:
                        nr_bins = int(max([np.max(sum_real), np.max(sum_fake)]) + 1)
                    pdf_real = np.zeros(nr_bins)
                    pdf_fake = np.zeros(nr_bins)
                    for real in sum_real:
                        if minimum < 0:
                            pdf_real[int(real - minimum)] += 1
                        else:
                            pdf_real[int(real)] += 1
                    for fake in sum_fake:
                        if minimum < 0:
                            pdf_fake[int(fake - minimum)] += 1
                        else:
                            pdf_fake[int(fake)] += 1
                    pdf_real /= len(sum_real)
                    pdf_fake /= len(sum_fake)
                    data_to_plot = [{'pdf': pdf_real, 'name': data[0]['name'], 'color': data[0]['color']},
                                    {'pdf': pdf_fake, 'name': data[1]['name'], 'color': data[1]['color']}]
                    if plot:
                        file = '{0}/attribute{1}_dim{2}_feature{3}'.format(dir, a, i, feature_dim)
                        if not os.path.exists(file):
                            os.makedirs(file)
                        file = "{0}/{1}".format(file, epoch)
                        plot_cdf(data=data_to_plot,
                                 file=file, normalization=feature.normalization)
                    d.append(wasserstein_distance(pdf_real, pdf_fake) - 1)
                    columns.append('attribute{}_dim{}_feature{}'.format(a, i, f))
            feature_dim += feature.dim
    if w_distance:
        m = statistics.mean(d)
        d.append(m)
        columns.append('Total average')
        d = np.asarray(d)
        d = np.reshape(d, (1, len(d)))
        columns = np.asarray(columns)
        df = pd.DataFrame(data=d, columns=columns)
        df.to_csv('{0}/w_distance_{1}.csv'.format(dir, epoch), sep=';')


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
    plt.savefig("{0}.png".format(file))


def nearest_neighbors(dir, real_data_features, sampled_data_features, data_feature_outputs):
    feature_dim = 0
    for f in range(len(data_feature_outputs)):
        if data_feature_outputs[f].type_ == output.OutputType.DISCRETE:
            feature_dim += data_feature_outputs[f].dim
            continue
        data_to_plot = []
        for i in range(3):
            # select random sample
            sample_nr = random.randint(0, sampled_data_features.shape[0] - 1)
            sample_feature = sampled_data_features[sample_nr, :, feature_dim]
            # calculate distance to all real samples
            dist = np.zeros(real_data_features.shape[0])
            for s in range(real_data_features.shape[0]):
                real_feature = real_data_features[s, :, feature_dim]
                # mse = ((sample_feature - real_feature)**2).mean()
                mse = metrics.mean_squared_error(real_feature, sample_feature)
                dist[s] = mse
            # get indices with lowest mse
            dist_ind = np.argsort(dist)
            data_to_plot.append({"sample_index": sample_nr, "nn_indices": dist_ind})
        file = "{0}/NN_feature_{1}".format(dir, f)
        plot_nearest_neighbors(data=data_to_plot, file=file,
                               real_data_features=real_data_features, sampled_data_features=sampled_data_features,
                               feature_dim=f)


### RESOURCE COSTS ###
def mean_square_error(x, y):
    return metrics.mean_squared_error(x, y)


# TODO: apparently average mse ?!
def mse_autocorrelation(dir, real_data_features, sample_data_features, data_feature_outputs):
    # create folder to save files
    dir = "{0}/mse_autocorrelation".format(dir)
    if not os.path.exists(dir):
        os.makedirs(dir)
    feature_dim = 0
    mses = []
    columns = []
    for f in range(len(data_feature_outputs)):
        if data_feature_outputs[f].type_ == output.OutputType.DISCRETE:
            feature_dim += data_feature_outputs[f].dim
            continue
        real_features = real_data_features[:, :, feature_dim]
        sample_features = sample_data_features[:, :, feature_dim]
        real_auto = np.apply_along_axis(func1d=autocorr, axis=1, arr=real_features)
        sample_auto = np.apply_along_axis(func1d=autocorr, axis=1, arr=sample_features)
        real_avg_auto = np.mean(real_auto, axis=0)
        sample_avg_auto = np.mean(sample_auto, axis=0)
        mse = metrics.mean_squared_error(real_avg_auto, sample_avg_auto)
        mses.append(mse)
        columns.append('feature_{0}'.format(f))
    mses = np.reshape(mses, (1, len(mses)))
    columns = np.asarray(columns)
    df = pd.DataFrame(data=mses, columns=columns)
    df.to_csv('{0}/mse_auto.csv'.format(dir), sep=';')


# load web dataset for testing
fake_data = True
w_lambert = False
kernel_smoothing = None

datasets = [{'name': "web", 'auto': [(50, False)]}]
"""
datasets = [
    {'name': "index_growth_range_1mo", 'auto': [(15, False), (8, True)]},
    {'name': "index_growth_range_3mo", 'auto': [(50, False), (20, True)]},
    {'name': "index_growth_range_12mo", 'auto': [(200, False), (150, True)]}
    ]
"""
eval_metrics = ['metadata', 'auto', 'NN', 'QQ', 'NN']
# eval_metrics = ['auto']
normalize = True
gan_types = ['Gen_TRANSFORMER_Dis_MLP']
embedding_metrics = ['TSNE', 'PCA']
qq_metrics = ['mean', 'variance', 'skewness', 'kurtosis']

for dataset in datasets:
    # load original data
    (data_feature, data_attribute, data_gen_flag, data_feature_outputs, data_attribute_outputs) = \
        load_data("data/{0}".format(dataset['name']))
    # if normalization needed
    if normalize:
        (data_feature, data_attribute, data_attribute_outputs, real_attribute_mask) = \
            normalize_per_sample(data_feature, data_attribute, data_feature_outputs, data_attribute_outputs,
                                 data_gen_flag)
    data = []
    # append real data
    data.append({
        'data_feature': data_feature,
        'data_attribute': data_attribute,
        'data_gen_flag': data_gen_flag,
        'color': 'yellow',
        'name': 'REAL'
    })
    eval_dir = "evaluation/{}".format(dataset['name'])
    if w_lambert:
        eval_dir = "{}_wl".format(eval_dir)
    if kernel_smoothing is not None:
        eval_dir = "{}_ks_{}".format(eval_dir, kernel_smoothing)
    if fake_data:
        for gan_type in gan_types:
            for i in range(1, 2):
                for n in range(0, 420, 20):
                    sample_path = 'runs/{}/{}/{}/checkpoint/epoch_{}/generated_samples.npz'.format(dataset['name'],
                                                                                                   gan_type, i, n)
                    sampled_data = np.load(sample_path)
                    sampled_features = sampled_data['sampled_features']
                    sampled_attributes = sampled_data['sampled_attributes']
                    sampled_gen_flags = sampled_data['sampled_gen_flags']
                    sampled_lengths = sampled_data['sampled_lengths']
                    # append sampled data
                    data.append({
                        'data_feature': sampled_features,
                        'data_attribute': sampled_attributes,
                        'data_gen_flag': sampled_gen_flags,
                        'data_lengths': sampled_lengths,
                        'color': 'blue',
                        'name': gan_type
                    })
                    dir = "{}/{}/{}/epoch_{}".format(eval_dir, gan_type, i, n)
                    if not os.path.exists(dir):
                        os.makedirs(dir)
                    # call methods
                    if 'metadata' in eval_metrics:
                        metadata_distribution(dir=dir, data=data, attribute_output=data_attribute_outputs)
                    if 'seq_len' in eval_metrics:
                        sequence_length(dir=dir, data=data)
                    if 'cross_meas' in eval_metrics:
                        cross_measurement(dir=dir, data=data, nr_bins=100)
                    if 'measurement' in eval_metrics:
                        measurement_distribution(dir=dir, data=data, feature_output=data_feature_outputs)
                    if 'EMD' in eval_metrics:
                        emd(dir=dir, data=data, data_feature_output=data_feature_outputs, epoch=n)
                    if 'auto' in eval_metrics:
                        for metric in dataset['auto']:
                            partial = metric[1]
                            if dataset['name'] == 'web' or dataset['name'] == 'FCC_MBA':
                                if partial:
                                    n_lags = (data[0]['data_feature'].shape[1] // 2) - 1
                                else:
                                    n_lags = data[0]['data_feature'].shape[1]
                            else:
                                n_lags = metric[0]
                            autocorrelation(dir=dir, data=data, data_feature_output=data_feature_outputs,
                                            n_lags=n_lags, partial=partial)
                    if 'NN' in eval_metrics:
                        nearest_neighbors(dir=dir, real_data_features=data_feature,
                                          sampled_data_features=sampled_features,
                                          data_feature_outputs=data_feature_outputs)
                    if "meta_meas" in eval_metrics:
                        meta_meas_corr(dir=dir, data=data, data_attribute_outputs=data_attribute_outputs,
                                       data_feature_outputs=data_feature_outputs)
                    if 'embedding' in eval_metrics:
                        for metric in embedding_metrics:
                            embedding(dir=dir, data=data, data_feature_output=data_feature_outputs,
                                      embedding=metric)
                    if 'QQ' in eval_metrics:
                        for metric in qq_metrics:
                            qq_plot(dir=dir, data=data, data_feature_output=data_feature_outputs, metric=metric)
                    data.pop(-1)
    else:
        dir = "{}/only_real".format(eval_dir)
        if not os.path.exists(dir):
            os.makedirs(dir)
        # call methods
        if 'metadata' in eval_metrics:
            metadata_distribution(dir=dir, data=data, attribute_output=data_attribute_outputs)
        if 'seq_len' in eval_metrics:
            sequence_length(dir=dir, data=data)
        if 'cross_meas' in eval_metrics:
            cross_measurement(dir=dir, data=data, nr_bins=100)
        if 'measurement' in eval_metrics:
            measurement_distribution(dir=dir, data=data, feature_output=data_feature_outputs)
        if 'auto' in eval_metrics:
            for metric in dataset['auto']:
                partial = metric[1]
                if dataset['name'] == 'web' or dataset['name'] == 'FCC_MBA':
                    if partial:
                        n_lags = (data[0]['data_feature'].shape[1] // 2) - 1
                    else:
                        n_lags = data[0]['data_feature'].shape[1]
                else:
                    n_lags = metric[0]
                autocorrelation(dir=dir, data=data, data_feature_output=data_feature_outputs,
                                n_lags=n_lags, partial=partial)
        # meta_meas_corr(dir=evaluation_dir, data=data, data_attribute_outputs=data_attribute_outputs,
        #                data_feature_outputs=data_feature_outputs)
        if 'embedding' in eval_metrics:
            for metric in embedding_metrics:
                embedding(dir=dir, data=data, data_feature_output=data_feature_outputs, embedding=metric)
