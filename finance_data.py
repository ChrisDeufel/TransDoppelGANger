import pandas as pd
import numpy as np
import os
from output import Output, OutputType, Normalization
import pickle


def calc_decade(year):
    if year < 1990:
        return 0
    elif year < 2000:
        return 1
    elif year < 2010:
        return 2
    elif year < 2020:
        return 3
    else:
        return 4


################# ATTRIBUTES #################
data_attribute_outputs = []
month = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  #
data_attribute_outputs.append(Output(type_=OutputType.DISCRETE, dim=len(month), normalization=None))
continent = [0, 1, 2, 3]  # [South America, North America, Europe, Asia]
data_attribute_outputs.append(Output(type_=OutputType.DISCRETE, dim=len(continent), normalization=None))
# type
decade = [0, 1, 2, 3, 4]  # [80's, 90's, 00's, 10's, 20's]
data_attribute_outputs.append(Output(type_=OutputType.DISCRETE, dim=len(decade), normalization=None))
################# FEATURES #################
data_feature_outputs = []
# growth
data_feature_outputs.append(Output(type_=OutputType.CONTINUOUS, dim=1, normalization=Normalization.ZERO_ONE))

max_seq_len = 25
attr_range = 0
for attr in data_attribute_outputs:
    attr_range += attr.dim
data_attribute = np.zeros((0, attr_range))
feat_range = 0
for feat in data_feature_outputs:
    feat_range += feat.dim
data_feature = np.zeros((0, max_seq_len, feat_range))
data_gen_flag = np.zeros((0, max_seq_len))

finance_path = "data/finance"
#csv_path = "{}/csvs".format(finance_path)
for root, dirs, files in os.walk(finance_path):
    for name in files:
        if name[-3:] == 'txt':
            continue
        df = pd.read_csv("{}/{}".format(finance_path, name), delimiter=";")
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        df['Open'] = df['Open'].str.replace(".", "")
        #df['Open'] = df['Open'].str.replace("0", "nan")
        df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
        #df['Open'] = df['Open'].replace(0.0, np.nan, inplace=True)
        df['Open'] = df['Open'].fillna(method='ffill')
        idx = 0
        while idx < len(df):
            current_row_idx = idx
            current_row = df.iloc[current_row_idx]
            # calculate sequence for one month
            current_row_month = current_row['Date'].month
            next_month = False
            while not next_month:
                if idx+1 < len(df):
                    next_row = df.iloc[idx + 1]
                else:
                    break
                next_row_month = next_row['Date'].month
                if current_row_month != next_row_month:
                    next_month = True
                else:
                    idx += 1
            seq = df.iloc[current_row_idx:idx]
            ### fill attributes for this row
            attribute = np.zeros(attr_range)
            # month
            attribute[current_row_month - 1] = 1
            # continent
            attribute[len(month) + current_row['Continent']] = 1
            # decade
            decade = calc_decade(current_row['Date'].year)
            attribute[len(month) + len(continent) + decade] = 1
            ### fill feature for this row
            feature = np.zeros((max_seq_len, feat_range))
            #current_course = np.asarray(seq['Growth'])
            current_course = np.asarray(seq['Open'])
            try:
                current_course = current_course.astype(np.float)*0.000001
                feature[:len(current_course), 0] = current_course
            except ValueError:
                idx += 1
                continue
            ### fill gen flag for this row
            gen_flag = np.zeros(max_seq_len)
            gen_flag[:len(current_course)] = 1
            ### concatenate to whole data
            data_attribute = np.concatenate((data_attribute, np.expand_dims(attribute, axis=0)), axis=0)
            data_feature = np.concatenate((data_feature, np.expand_dims(feature, axis=0)), axis=0)
            data_gen_flag = np.concatenate((data_gen_flag, np.expand_dims(gen_flag, axis=0)), axis=0)
            idx += 1
        print("{} finished - current len: {}".format(name, len(data_attribute)))

path = "data/index_course"
dbfile = open('{}/data_attribute_output.pkl'.format(path), 'ab')
pickle.dump(data_attribute_outputs, dbfile)
dbfile.close()
dbfile = open('{}/data_feature_output.pkl'.format(path), 'ab')
pickle.dump(data_feature_outputs, dbfile)
dbfile.close()

np.savez("{}/data_train.npz".format(path), data_attribute=data_attribute, data_feature=data_feature, data_gen_flag=data_gen_flag)

# df = pd.read_csv(finance_path)
