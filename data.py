from torch.utils.data import Dataset
import torch
from load_data import load_data
from util import add_gen_flag, normalize_per_sample

class Data(Dataset):
    def __init__(self, sample_len, transforms=None, normalize=True, gen_flag=True, name='web'):
        (self.data_feature, self.data_attribute, self.data_gen_flag,
         self.data_feature_outputs, self.data_attribute_outputs) = load_data("data/{0}".format(name))
        self.transforms = transforms
        self.name = name

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
