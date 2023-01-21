import pandas as pd
from output import Output, OutputType, Normalization
import numpy as np
import pickle
import os
from random import randrange

min_seq_len = 800
max_seq_len = 1000
max_samples = 2000
min_samples = 1500

################# ATTRIBUTES #################
data_attribute_outputs = []
countries = ["us", "uk", "hk", "pl", "jp"]  # [0, 1, 2, 3]
# countries = ["pl"]
data_attribute_outputs.append(Output(type_=OutputType.DISCRETE, dim=len(countries), normalization=None))

data_attribute = np.zeros((0, len(countries)))
################# FEATURES #################
data_feature_outputs = []



data_gen_flag = np.zeros((0, max_seq_len))

path = "data/indices"
col_1 = "open"
data_feature_outputs.append(Output(type_=OutputType.CONTINUOUS, dim=1, normalization=Normalization.ZERO_ONE))
col_2 = "growth"
data_feature_outputs.append(Output(type_=OutputType.CONTINUOUS, dim=1, normalization=Normalization.MINUSONE_ONE))
col_3 = "range"
data_feature_outputs.append(Output(type_=OutputType.CONTINUOUS, dim=1, normalization=Normalization.ZERO_ONE))
data_feature = np.zeros((0, max_seq_len, 3))

for i in range(len(countries)):
    country_path = os.path.join(path, countries[i])
    counter = 0
    for (dirpath, dirnames, filenames) in os.walk(country_path):
        for file in filenames:
            file_path = os.path.join(country_path, file)
            try:
                in_df = pd.read_csv(file_path)
            except:
                print("{} couldnt be loaded".format(file_path))
                continue
            if len(in_df) > min_seq_len:
                df_len = randrange(min_seq_len, max_seq_len)
                in_df = in_df.head(df_len)
            else:
                continue
            in_df = in_df.dropna()
            in_df[col_1] = in_df["<OPEN>"]
            in_df[col_2] = (in_df["<CLOSE>"] - in_df["<OPEN>"]) / in_df["<OPEN>"]
            in_df[col_3] = in_df["<HIGH>"] - in_df["<LOW>"]
            in_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            if in_df.isnull().values.any() or len(in_df[col_2].unique()) == 1:
                continue


            # attribute
            in_attribute = np.zeros(len(countries))
            in_attribute[i] = 1
            in_attribute = np.expand_dims(in_attribute, axis=0)
            data_attribute = np.concatenate((data_attribute, in_attribute), axis=0)
            # feature
            price = np.asarray(in_df[col_1])
            feature = np.zeros((max_seq_len, 3))
            feature[:len(price), 0] = price

            growth = np.asarray(in_df[col_2])
            feature[:len(price), 1] = growth

            range = np.asarray(in_df[col_3])
            feature[:len(price), 2] = range

            feature = np.expand_dims(feature, axis=(0))
            data_feature = np.concatenate((data_feature, feature), axis=0)
            # gen flag
            gen_flag = np.zeros(max_seq_len)
            gen_flag[:len(price)] = 1
            data_gen_flag = np.concatenate((data_gen_flag, np.expand_dims(gen_flag, axis=0)), axis=0)
            counter += 1
        print("{} finished - {}".format(countries[i], counter))

path = path + "_{}_{}_{}_{}".format(col_1, col_2, col_3, max_seq_len)
if not os.path.exists(path):
    os.makedirs(path)
dbfile = open('{}/data_attribute_output.pkl'.format(path), 'ab')
pickle.dump(data_attribute_outputs, dbfile)
dbfile.close()
dbfile = open('{}/data_feature_output.pkl'.format(path), 'ab')
pickle.dump(data_feature_outputs, dbfile)
dbfile.close()

np.savez("{}/data_train.npz".format(path), data_attribute=data_attribute, data_feature=data_feature,
         data_gen_flag=data_gen_flag)
