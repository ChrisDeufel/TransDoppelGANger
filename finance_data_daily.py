import yfinance as yf
import pandas as pd
import os
from output import Output, OutputType, Normalization
import pickle
import datetime
import numpy as np
import logging

measurement = "growth"

################# ATTRIBUTES #################
data_attribute_outputs = []
continent = [0, 1, 2, 3]    # [South America, North America, Europe, Asia]
data_attribute_outputs.append(Output(type_=OutputType.DISCRETE, dim=len(continent), normalization=None))
################# FEATURES #################
data_feature_outputs = []
# course
if measurement == "price":
    norm = Normalization.ZERO_ONE
else:
    norm = Normalization.MINUSONE_ONE
data_feature_outputs.append(Output(type_=OutputType.CONTINUOUS, dim=1, normalization=norm))

end_date = datetime.date.today()
start_date = datetime.date(year=end_date.year, day=end_date.day, month=end_date.month-2)
delta = datetime.timedelta(days=1)
intervall = "1m"


max_seq_len = 400
attr_range = 0
for attr in data_attribute_outputs:
    attr_range += attr.dim
data_attribute = np.zeros((0, attr_range))
feat_range = 0
for feat in data_feature_outputs:
    feat_range += feat.dim
data_feature = np.zeros((0, max_seq_len, feat_range))
data_gen_flag = np.zeros((0, max_seq_len))

indices = [{"ticker": "^GSPC", "continent": 1},
           {"ticker": "^FTSE", "continent": 2},
           {"ticker": "^N225", "continent": 3},
           {"ticker": "^BVSP", "continent": 0},
           {"ticker": "^GDAXI", "continent": 2},
           {"ticker": "^000001.SS", "continent": 3},
           {"ticker": "^IXIC", "continent": 1},
           {"ticker": "^MERV", "continent": 0}]

path = "data/index_{}_{}".format(measurement, str(intervall))
if not os.path.exists(path):
    os.makedirs(path)
f = open("{}/description.txt".format(path), "a")
f.write("DESCRIPTION OF THE FINANCE DATASET")
f.write("Measurement: {}".format(measurement))

for index in indices:
    start_date = datetime.date(year=end_date.year, day=end_date.day, month=end_date.month - 2)
    # iterate over all dates
    while start_date < end_date:
        raw_data = yf.download(tickers=index['ticker'],
                               start=str(start_date),
                               end=str(start_date + delta),
                               interval="1m",
                               group_by='ticker',
                               auto_adjust=True,
                               treads=True)
        if len(raw_data) < (max_seq_len/2):
            start_date += delta
            continue
        raw_data['Open'] = raw_data['Open'].fillna(method='ffill')
        raw_data['Growth'] = (raw_data['Close']-raw_data['Open'])/raw_data['Open']
        # add attributes
        # continent
        cont = np.zeros(len(continent))
        cont[index["continent"]] = 1
        attribute = np.expand_dims(cont, axis=0)
        data_attribute = np.concatenate((data_attribute, attribute), axis=0)
        # add features
        # course
        feature = np.zeros((max_seq_len, 1))
        if measurement == "price":
            current_course = np.asarray(raw_data['Open'])
        else:
            current_course = np.asarray(raw_data['Growth'])
        feature[:len(current_course), 0] = current_course
        feature = np.expand_dims(feature, axis=0)
        data_feature = np.concatenate((data_feature, feature), axis=0)
        # add gen flag
        gen_flag = np.zeros(max_seq_len)
        gen_flag[:len(current_course)] = 1
        data_gen_flag = np.concatenate((data_gen_flag, np.expand_dims(gen_flag, axis=0)), axis=0)
        start_date += delta
    print("{} finished - current len: {}".format(index['ticker'], len(data_attribute)))

if not os.path.exists(path):
    os.makedirs(path)
dbfile = open('{}/data_attribute_output.pkl'.format(path), 'ab')
pickle.dump(data_attribute_outputs, dbfile)
dbfile.close()
dbfile = open('{}/data_feature_output.pkl'.format(path), 'ab')
pickle.dump(data_feature_outputs, dbfile)
dbfile.close()

np.savez("{}/data_train.npz".format(path), data_attribute=data_attribute, data_feature=data_feature, data_gen_flag=data_gen_flag)
