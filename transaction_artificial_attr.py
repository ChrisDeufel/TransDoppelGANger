import numpy as np
import pickle
import os
import random
from output import Output, OutputType, Normalization


path = "data/transactions/"
attribute_output_path = "data/transactions/data_attribute_output.pkl"
feature_output_path = "data/transactions/data_feature_output.pkl"

with open(os.path.join(path, "data_feature_output.pkl"), "rb") as f:
    data_feature_outputs = pickle.load(f)
with open(os.path.join(path, "data_attribute_output.pkl"), "rb") as f:
    data_attribute_outputs = pickle.load(f)
# determine min and max of amounts
min_amount = 100000000
max_amount = -100000000
for idx in range(2000):
    feature_path = path + "{}_data_feature.npy".format(idx)
    features = np.load(feature_path)
    total_amount = np.sum(features[idx, 0])
    if total_amount < min_amount:
        min_amount = total_amount
    if total_amount > max_amount:
        max_amount = total_amount

amount_range = max_amount - min_amount
# generate artificial attribute age
nr_age_groups = 4
for idx in range(2000):
    feature_path = path + "{}_data_feature.npy".format(idx)
    features = np.load(feature_path)
    total_amount = np.sum(features[idx, 0])
    total_amount_norm = total_amount - min_amount

    # generate new attribute
    attribute = np.zeros(nr_age_groups)
    if random.uniform(0, 1) < 0.25:
        id = int(random.uniform(0, nr_age_groups))
    else:
        id = int((total_amount_norm/amount_range)*nr_age_groups)
    if id == nr_age_groups:
        attribute[id-1] = 1
    else:
        attribute[id] = 1
    attribute_path = path + "{}_data_attribute.npy".format(idx)
    attributes = np.save(attribute_path, attribute)

data_attribute_outputs = [Output(type_=OutputType.DISCRETE, dim=nr_age_groups, normalization=None)]
dbfile = open('data/transactions/data_attribute_output.pkl', 'ab')
# source, destination
pickle.dump(data_attribute_outputs, dbfile)
dbfile.close()
print('hello')