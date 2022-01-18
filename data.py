from torch.utils.data import Dataset
import torch
from load_data import load_data
from util import add_gen_flag, add_gen_flag_split, normalize_per_sample, normalize_per_sample_split, extract_len, \
    NormMinMax
import numpy as np
import os
import pickle
import copy


class Data(Dataset):
    def __init__(self, sample_len, transforms=None, normalize=True, gen_flag=True, size=None, name='web'):
        (self.data_feature, self.data_attribute, self.data_gen_flag,
         self.data_feature_outputs, self.data_attribute_outputs) = load_data("data/{0}".format(name))
        self.transforms = transforms
        self.name = name
        if size is not None:
            x = np.arange(len(self.data_attribute))
            rand = np.random.choice(x, size=size, replace=False)
            self.data_feature = self.data_feature[rand, :, :]
            self.data_attribute = self.data_attribute[rand, :]
            self.data_gen_flag = self.data_gen_flag[rand, :]
        if normalize:
            (self.data_feature, self.data_attribute, self.data_attribute_outputs,
             self.real_attribute_mask) = normalize_per_sample(self.data_feature, self.data_attribute,
                                                              self.data_feature_outputs, self.data_attribute_outputs,
                                                              self.data_gen_flag)

        if gen_flag:
            self.data_feature, self.data_feature_outputs = add_gen_flag(self.data_feature, self.data_gen_flag,
                                                                        self.data_feature_outputs, sample_len)
        self.data_feature_shape = self.data_feature.shape
        self.data_attribute_shape = self.data_attribute.shape
        np.save("data/{}/data_feature_n_g.npy".format(name), self.data_feature)
        np.save("data/{}/data_attribute_n_g.npy".format(name), self.data_attribute)

    def __getitem__(self, idx):
        attribute = torch.Tensor(self.data_attribute[idx, :])
        feature = torch.Tensor(self.data_feature[idx, :, :])
        if self.transforms:
            attribute = self.transforms(attribute)
            feature = self.transforms(feature)
        return attribute, feature

    def __len__(self):
        return len(self.data_attribute)


class LargeData(Dataset):
    def __init__(self, sample_len, nr_samples, transforms=None, normalize=True, gen_flag=True, splits=10,
                 name='google_split'):
        self.transforms = transforms
        self.splits = int(splits)
        self.nr_samples = nr_samples * self.splits
        self.split_size = int(self.nr_samples / self.splits)
        self.name = name
        self.normalize = normalize
        self.gen_flag = gen_flag
        self.sample_len = sample_len

    def __getitem__(self, idx):
        split_nr = int(idx // self.split_size)
        idx = int(idx % self.split_size)
        split_path = "data/google_split/w_normalizationAndGenFlag/data_train_{0}.npz".format(split_nr)
        data_npz = np.load(split_path)
        data_feature = data_npz["data_feature"]
        data_attribute = data_npz["data_attribute"]
        attribute = torch.Tensor(data_attribute[idx, :])
        feature = torch.Tensor(data_feature[idx, :, :])
        if self.transforms:
            attribute = self.transforms(attribute)
            feature = self.transforms(feature)
        return attribute, feature

    def __len__(self):
        return self.nr_samples


class SplitData(Dataset):
    def __init__(self, sample_len, transforms=None, normalize=True, gen_flag=True, size=None, name='transactions'):
        # count number of samples for this dataset
        self.normalize = normalize
        self.gen_flag = gen_flag
        self.data_path = "data/{}".format(name)
        self.nr_samples = 0
        for item in os.listdir(self.data_path):
            if "feature.npy" in item:
                self.nr_samples += 1
        self.nr_samples = int(self.nr_samples)
        # load output files and gen flags
        self.data_gen_flag = np.load("data/{}/data_gen_flag.npy".format(name))
        with open(os.path.join(self.data_path, "data_feature_output.pkl"), "rb") as f:
            self.data_feature_outputs = pickle.load(f)
        with open(os.path.join(self.data_path,
                               "data_attribute_output.pkl"), "rb") as f:
            self.data_attribute_outputs = pickle.load(f)
        if normalize:
            if not os.path.exists("data/{}/normalized".format(name)):
                self.data_attribute_outputs, self.real_attribute_mask = normalize_per_sample_split(path=self.data_path,
                                                                                                   nr_samples=self.nr_samples,
                                                                                                   data_feature_outputs=self.data_feature_outputs,
                                                                                                   data_attribute_outputs=self.data_attribute_outputs)
            else:
                with open("{}/normalized/data_feature_output.pkl".format(self.data_path),
                          "rb") as f:
                    self.data_feature_outputs = pickle.load(f)
                with open("{}/normalized/data_attribute_output.pkl".format(self.data_path),
                          "rb") as f:
                    self.data_attribute_outputs = pickle.load(f)
                self.real_attribute_mask = np.load("{}/normalized/real_attribute_mask.npy".format(self.data_path))
            self.data_path = "{}/normalized".format(self.data_path)
        if gen_flag:
            if not os.path.exists("{}/gen_flag".format(self.data_path)):
                self.data_feature_outputs = add_gen_flag_split(self.data_path, self.nr_samples, self.data_gen_flag,
                                                               self.data_feature_outputs, sample_len)
            else:
                with open("{}/gen_flag/data_feature_output.pkl".format(self.data_path), "rb") as f:
                    self.data_feature_outputs = pickle.load(f)

        ################ WALKAROUND TO PROVIDE VARIABLES NECESSARY FOR ARCHITECTURES ################
        # TODO: find other way to provide variables
        if self.normalize and self.gen_flag:
            self.rand_data_feature = np.load("{}/gen_flag/0_data_feature.npy".format(self.data_path))
            self.rand_data_attribute = np.load("{}/0_data_attribute.npy".format(self.data_path))
        self.data_feature_shape = (self.nr_samples, self.rand_data_feature.shape[0], self.rand_data_feature.shape[1])
        self.data_attribute_shape = (self.nr_samples, self.rand_data_attribute.shape[0])

    def __getitem__(self, idx):
        if self.normalize and self.gen_flag:
            feature = np.load("{}/gen_flag/{}_data_feature.npy".format(self.data_path, idx))
            attribute = np.load("{}/{}_data_attribute.npy".format(self.data_path, idx))
        return torch.Tensor(attribute), torch.Tensor(feature)

    def __len__(self):
        return self.nr_samples


class DCData(Dataset):
    def __init__(self, real_feature, fake_feature):
        self.features = np.concatenate((real_feature, fake_feature), axis=0)
        # fake = 0
        fake_labels = np.zeros(len(fake_feature))
        # real = 1
        real_labels = np.ones(len(real_feature))
        self.labels = np.concatenate((real_labels, fake_labels))

    def __getitem__(self, idx):
        feature = self.features[idx, :, :]
        label = self.labels[idx]
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.features)


class TimeGanData(Dataset):
    def __init__(self, transforms=None, normalize=False, gen_flag=True, size=None, name='web'):
        (self.data_feature, self.data_attribute, self.data_gen_flag,
         self.data_feature_outputs, self.data_attribute_outputs) = load_data("data/{0}".format(name))
        self.transforms = transforms
        self.ori_data, self.min_val, self.max_val = NormMinMax(self.data_feature)
        self.lengths, self.max_seq_len = extract_len(self.data_gen_flag)
        self.lengths = np.expand_dims(self.lengths, axis=1)
        self.name = name
        if size is not None:
            x = np.arange(len(self.data_attribute))
            rand = np.random.choice(x, size=size, replace=False)
            self.data_feature = self.data_feature[rand, :, :]
            self.data_attribute = self.data_attribute[rand, :]
            self.data_gen_flag = self.data_gen_flag[rand, :]
        if normalize:
            (self.data_feature, self.data_attribute, self.data_attribute_outputs,
             self.real_attribute_mask) = normalize_per_sample(self.data_feature, self.data_attribute,
                                                              self.data_feature_outputs, self.data_attribute_outputs,
                                                              self.data_gen_flag)

        if gen_flag:
            self.data_feature, self.data_feature_outputs = add_gen_flag(self.data_feature, self.data_gen_flag,
                                                                        self.data_feature_outputs, sample_len)
        self.data_feature_shape = self.data_feature.shape
        self.data_attribute_shape = self.data_attribute.shape

    def __getitem__(self, idx):
        length = torch.tensor(self.lengths[idx], dtype=torch.float32)
        feature = torch.tensor(self.data_feature[idx, :, :], dtype=torch.float32)
        if self.transforms:
            length = self.transforms(length)
            feature = self.transforms(feature)
        return length, feature

    def __len__(self):
        return len(self.data_feature)