import math
import statistics
import random
from scipy import stats
from sklearn import metrics
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import statsmodels.api as sm
import output
from load_data import load_data
from util import add_gen_flag, normalize_per_sample

sys.modules["output"] = output


### AUTOCORRELATION ###
def autocorr(x):
    nlags = len(x)
    autocorr = sm.tsa.acf(x, nlags=nlags)
    return autocorr


def plot_auto(data_real, data_fake):
    assert len(data_real) == len(data_fake)
    for i in range(len(data_real)):
        assert data_real[i]["file"] == data_fake[i]["file"]
        fig, axes = plt.subplots(1, 1, figsize=(12, 4))
        axes.plot(data_real[i]["auto_data"], color=data_real[i]["color"], label=data_real[i]["name"])
        axes.plot(data_fake[i]["auto_data"], color=data_fake[i]["color"], label=data_fake[i]["name"])
        axes.legend()
        axes.set_title("Autocorrelation")
        file = data_real[i]["file"]
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
    # dir = "{0}/{1}".format(dir, epoch)
    feature_dim = 0
    data_to_plot = []
    for f in data_feature_output:
        if f.type_ == output.OutputType.DISCRETE:
            feature_dim += f.dim
            continue
        data_feature = data['data_feature']
        data_feature = data_feature[:, :, feature_dim]
        auto = np.apply_along_axis(func1d=autocorr, axis=1, arr=data_feature)
        data_avg_auto = np.mean(auto, axis=0)
        file = "{0}/feature_{1}".format(dir, feature_dim)
        if not os.path.exists(file):
            os.makedirs(file)
        file = "{0}/{1}".format(file, epoch)
        data_to_plot.append({"auto_data": data_avg_auto, "name": data['name'], "color": data['color'], "file": file})
        feature_dim += f.dim
    return data_to_plot


### SEQUENCE LENGTH ###
def plot_seq_len(data_real, data_fake):
    assert len(data_real) == len(data_fake)
    for i in range(len(data_real)):
        assert data_real[i]["file"] == data_fake[i]["file"]
        fig, axes = plt.subplots(1, 1, figsize=(12, 4))
        data_bins = data_real[i]['data_bins']
        x = np.arange(start=0, stop=len(data_bins), step=1)
        axes.bar(x, data_bins, color=data_real[i]['color'], label=data_real[i]['name'], alpha=0.5)
        data_bins = data_fake[i]['data_bins']
        x = np.arange(start=0, stop=len(data_bins), step=1)
        axes.bar(x, data_bins, color=data_fake[i]['color'], label=data_fake[i]['name'], alpha=0.5)
        axes.legend()
        axes.set_title("Sequence Length")
        plt.savefig('{0}.png'.format(data_real[i]["file"]))


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
    data_gen_flag = data["data_gen_flag"]
    len = np.count_nonzero(data_gen_flag, axis=1)
    data_bins = np.bincount(len)[:50]
    file = "{0}/epoch_{1}".format(dir, epoch)
    data_to_plot.append({"data_bins": data_bins, "name": data['name'], "color": data['color'], "file": file})
    return data_to_plot


### PEARSON CORRELATION ###
# calculate pearson coefficient
def pearson(x, y):
    return stats.pearsonr(x, y)


# calculate and plot cdf from list
def plot_cdf(data_real, data_fake):
    assert len(data_real) == len(data_fake)
    for i in range(len(data_real)):
        assert data_real[i]["file"] == data_fake[i]["file"]
        fig, axes = plt.subplots(1, 1, figsize=(12, 8))
        # plot real
        pdf = data_real[i]['pdf']
        Y = np.cumsum(pdf)
        X = np.arange(pdf.shape[0])
        axes.plot(X, Y, color=data_real[i]['color'], label=data_real[i]['name'])
        # plot fake
        pdf = data_fake[i]['pdf']
        Y = np.cumsum(pdf)
        X = np.arange(pdf.shape[0])
        axes.plot(X, Y, color=data_fake[i]['color'], label=data_fake[i]['name'])
        axes.legend()
        axes.set_title("CDF")
        file = data_real[i]['file']
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
    # create folder to save files
    eval_dir = dir.rsplit("/", 1)[0]
    epoch = dir.rsplit("/", 1)[1]
    dir = "{0}/{1}".format(eval_dir, "cross_meas_correlation")
    if not os.path.exists(dir):
        os.makedirs(dir)
    nr_features = data['data_feature'].shape[2]
    data_to_plot = []
    for f_1 in range(nr_features):
        for f_2 in range(f_1 + 1, nr_features):
            data_feature = data['data_feature']
            data_gen_flag = data['data_gen_flag']
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
            pearsons /= data_feature.shape[0]
            file = "{0}/feature{1}_feature{2}".format(dir, f_1, f_2)
            if not os.path.exists(file):
                os.makedirs(file)
            file = "{0}/{1}".format(file, epoch)
            data_to_plot.append({"pdf": pearsons, 'name': data['name'], 'color': data['color'], 'file': file})
    return data_to_plot


### PLOT MEASUREMENT DISTRIBUTION OR METADATA DISTRIBUTION ###
def plot_distribution(data_real, data_fake, title):
    assert len(data_real) == len(data_fake)
    for i in range(len(data_real)):
        assert data_real[i]["file"] == data_fake[i]["file"]
        file = data_real[i]["file"]
        fig, axes = plt.subplots(1, 1, figsize=(12, 4))
        bins = None
        bins = data_real[i]['bins']
        y_pos = np.arange(len(bins))
        axes.bar(y_pos, bins, color=data_real[i]['color'], label=data_real[i]['name'], alpha=0.5)

        bins = data_fake[i]['bins']
        y_pos = np.arange(len(bins))
        axes.bar(y_pos, bins, color=data_fake[i]['color'], label=data_fake[i]['name'], alpha=0.5)

        axes.legend()
        axes.set_title(title)
        if 'normalization' in data_real[i]:
            normalization = data_real[i]['normalization']
        else:
            normalization = None
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
    # dir = "{0}/{1}".format(dir, epoch)
    dim = 0
    counter = 0
    data_to_plot = []
    for f in feature_output:
        if f.type_ == output.OutputType.DISCRETE:
            dim += f.dim
            continue
        data_feature = data['data_feature']
        bins = np.zeros(nr_bins)
        for j in range(len(data_feature)):
            feature = data_feature[j, :, dim]
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

        dim += f.dim
        file = "{0}/feature_{1}".format(dir, counter)
        if not os.path.exists(file):
            os.makedirs(file)
        file = "{0}/{1}".format(file, epoch)
        data_to_plot.append({'bins': bins, 'name': data['name'], 'color': data['color'], 'file': file,
                             'normalization': f.normalization})
        counter += 1
    return data_to_plot


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
    data_to_plot = []
    for i in attribute_output:
        dim_range = dim + i.dim
        data_attribute = data['data_attribute']
        data_attribute = data_attribute[:, dim:dim_range]
        indices = np.argmax(data_attribute, axis=1)
        one_hot = np.zeros(data_attribute.shape)
        one_hot[np.arange(indices.size), indices] = 1
        data_attribute = one_hot
        bins = np.sum(data_attribute, axis=0)
        file = "{0}/attribute_{1}".format(dir, counter)
        if not os.path.exists(file):
            os.makedirs(file)
        file = "{0}/{1}".format(file, epoch)
        data_to_plot.append({'bins': bins, 'name': data['name'], 'color': data['color'], 'file': file})
        dim += i.dim
        counter += 1
    return data_to_plot


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
                                 file=file)
                    d.append(wasserstein_distance(pdf_real, pdf_fake) - 1)
                    columns.append('attribute{1}_dim{2}_feature{3}'.format(dataset, a, i, f))
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
    # create folder to save files
    eval_dir = dir.rsplit("/", 1)[0]
    epoch = dir.rsplit("/", 1)[1]
    dir = "{0}/{1}".format(eval_dir, "nearest_neighbor")
    if not os.path.exists(dir):
        os.makedirs(dir)
    # dir = "{0}/{1}".format(dir, epoch)
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
        file = "{0}/feature_{1}".format(dir, f)
        if not os.path.exists(file):
            os.makedirs(file)
        file = "{0}/{1}".format(file, epoch)
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


for n in range(1, 2, 1):
    for i in range(0, 500, 20):
        # load web dataset for testing
        dataset = 'google_m'

        # load original data
        (data_feature, data_attribute, data_gen_flag, data_feature_outputs, data_attribute_outputs) = \
            load_data("data/{0}".format(dataset))

        # if normalization needed
        # (data_feature, data_attribute, data_attribute_outputs, real_attribute_mask) = \
        #    normalize_per_sample(data_feature, data_attribute, data_feature_outputs, data_attribute_outputs)

        sample_path = 'runs/{}/RNN/{}/checkpoint/epoch_{}/generated_samples.npz'.format(dataset, n, i)
        # sample_path = 'runs/google_m/RNN/1/checkpoint/epoch_20/generated_samples.npz'

        # create folder to save data
        run_dir = "{}/{}".format(sample_path.split("/")[2], sample_path.split("/")[3])
        epoch_id = sample_path.split("/")[5]
        evaluation_dir = "evaluation/{0}/{1}".format(dataset, run_dir)
        if not os.path.exists(evaluation_dir):
            os.makedirs(evaluation_dir)
        evaluation_dir = "{0}/{1}".format(evaluation_dir, epoch_id)
        data = {
            'data_feature': data_feature,
            'data_attribute': data_attribute,
            'data_gen_flag': data_gen_flag,
            'color': 'yellow',
            'name': 'REAL'
        }
        # generate real data eval
        # real_data_to_plot = autocorrelation(dir=evaluation_dir, data=data, data_feature_output=data_feature_outputs)
        # real_data_to_plot = sequence_length(dir=evaluation_dir, data=data)
        # real_data_to_plot = measurement_distribution(dir=evaluation_dir, data=data, feature_output=data_feature_outputs)
        # real_data_to_plot = metadata_distribution(dir=evaluation_dir, data=data, attribute_output=data_attribute_outputs)
        real_data_to_plot = cross_measurement(dir=evaluation_dir, data=data, nr_bins=100)
        data = None
        data_feature = None
        data_attribute = None
        # load generated data
        sampled_data = np.load(sample_path)
        sampled_features = sampled_data['sampled_features']
        sampled_attributes = sampled_data['sampled_attributes']
        sampled_gen_flags = sampled_data['sampled_gen_flags']
        sampled_lengths = sampled_data['sampled_lengths']
        # append sampled data
        data = {
            'data_feature': sampled_features,
            'data_attribute': sampled_attributes,
            'data_gen_flag': sampled_gen_flags,
            'data_lengths': sampled_lengths,
            'color': 'blue',
            'name': 'torchDoppelGANger'
        }
        # fake_data_to_plot = autocorrelation(dir=evaluation_dir, data=data, data_feature_output=data_feature_outputs)
        # fake_data_to_plot = sequence_length(dir=evaluation_dir, data=data)
        # fake_data_to_plot = measurement_distribution(dir=evaluation_dir, data=data, feature_output=data_feature_outputs)
        # fake_data_to_plot = metadata_distribution(dir=evaluation_dir, data=data,
        #                                           attribute_output=data_attribute_outputs)
        fake_data_to_plot = cross_measurement(dir=evaluation_dir, data=data, nr_bins=100)
        # plot_auto(data_real=real_data_to_plot, data_fake=fake_data_to_plot)
        # plot_seq_len(data_real=real_data_to_plot, data_fake=fake_data_to_plot)
        # plot_distribution(data_real=real_data_to_plot, data_fake=fake_data_to_plot, title="measurement_distribution")
        # plot_distribution(data_real=real_data_to_plot, data_fake=fake_data_to_plot, title="metadata_distribution")
        plot_cdf(data_real=real_data_to_plot, data_fake=fake_data_to_plot)

        # call methods
        # metadata_distribution(dir=evaluation_dir, data=data, attribute_output=data_attribute_outputs)
        # sequence_length(dir=evaluation_dir, data=data)
        # cross_measurement(dir=evaluation_dir, data=data, nr_bins=100)
        # measurement_distribution(dir=evaluation_dir, data=data, feature_output=data_feature_outputs)
        # autocorrelation(dir=evaluation_dir, data=data, data_feature_output=data_feature_outputs)
        # nearest_neighbors(dir=evaluation_dir, real_data_features=data_feature, sampled_data_features=sampled_features,
        #                   data_feature_outputs=data_feature_outputs)
        # meta_meas_corr(dir=evaluation_dir, data=data, data_attribute_outputs=data_attribute_outputs,
        #               data_feature_outputs=data_feature_outputs)

        # mse_autocorrelation(dir=evaluation_dir, real_data_features=data_feature, sample_data_features=sampled_features,
        #                    data_feature_outputs=data_feature_outputs)
