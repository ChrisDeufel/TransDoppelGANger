from torch.utils.data import Dataset
import torch
from load_data import load_data
from util import add_gen_flag, normalize_per_sample
import numpy as np
import os
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
                                                              self.data_feature_outputs, self.data_attribute_outputs)

        if gen_flag:
            self.data_feature, self.data_feature_outputs = add_gen_flag(self.data_feature, self.data_gen_flag,
                                                                        self.data_feature_outputs, sample_len)

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
        self.nr_samples = nr_samples*self.splits
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


