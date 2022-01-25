import yfinance as yf
import pandas as pd
import os
from output import Output, OutputType, Normalization
import pickle
import datetime
import numpy as np
import logging
# from evaluation import measurement_distribution, autocorrelation
import matplotlib.pyplot as plt
from scipy.special import lambertw


def calc_decade(year):
    if year < 2000:
        return 0
    elif year < 2010:
        return 1
    else:
        return 2


measurement = "growth"
log = False
normalize_growth = "minusone_one"
normalize_range = "zero_one"
normalization = False
add_range = True

################# ATTRIBUTES #################
data_attribute_outputs = []
continent = ["Rest of the World", "North America", "Europe", "Asia"]  # [0, 1, 2, 3]
data_attribute_outputs.append(Output(type_=OutputType.DISCRETE, dim=len(continent), normalization=None))
decade = ["90s", "00s", "10s"]  # [0, 1, 2]
data_attribute_outputs.append(Output(type_=OutputType.DISCRETE, dim=len(decade), normalization=None))
################# FEATURES #################
data_feature_outputs = []
# course
if measurement == "price":
    norm = Normalization.ZERO_ONE
else:
    norm = Normalization.MINUSONE_ONE
data_feature_outputs.append(Output(type_=OutputType.CONTINUOUS, dim=1, normalization=norm))
# range
if add_range:
    data_feature_outputs.append(Output(type_=OutputType.CONTINUOUS, dim=1, normalization=Normalization.ZERO_ONE))
start_year = 1990
end_year = 2021
months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
m_intervall = 12

max_seq_len = 25 * m_intervall
attr_range = 0
for attr in data_attribute_outputs:
    attr_range += attr.dim
data_attribute = np.zeros((0, attr_range))
feat_range = 0
for feat in data_feature_outputs:
    feat_range += feat.dim
data_feature = np.zeros((0, max_seq_len, feat_range))
data_gen_flag = np.zeros((0, max_seq_len))
indices = [{"ticker": "^GSPC", "continent": 1, "country": "USA"}]
"""
indices = [{"ticker": "^GSPC", "continent": 1, "country": "USA"},
           {"ticker": "^FTSE", "continent": 2, "country": "UK"},
           {"ticker": "^N225", "continent": 3, "country": "JAPAN"},
           {"ticker": "^BVSP", "continent": 0, "country": "BRAZIL"},
           {"ticker": "^GDAXI", "continent": 2, "country": "GERMANY"},
           {"ticker": "^000001.SS", "continent": 3, "country": "CHINA"},
           {"ticker": "^IXIC", "continent": 1, "country": "USA"},
           {"ticker": "^MERV", "continent": 0, "country": "ARGENTINA"},
           {"ticker": "^BSESN", "continent": 3, "country": "INDIA"},
           {"ticker": "^FCHI", "continent": 2, "country": "FRANCE"},
           {"ticker": "^MXX", "continent": 0, "country": "MEXICO"},
           {"ticker": "^SPCOSLCP", "continent": 0, "country": "COLOMBIA"},
           {"ticker": "^HSI", "continent": 3, "country": "CHINA"},
           {"ticker": "^IPSA", "continent": 0, "country": "CHILE"},
           {"ticker": "^IBEX", "continent": 2, "country": "SPAIN"},
           {"ticker": "^AXJO", "continent": 0, "country": "AUSTRALIA"},
           {"ticker": "^KS11", "continent": 3, "country": "SOUTH KOREA"},
           {"ticker": "^DJI", "continent": 1, "country": "USA"},
           {"ticker": "^SET.BK", "continent": 3, "country": "THAILAND"},
           {"ticker": "^STI", "continent": 3, "country": "SINGAPORE"},
           {"ticker": "^OMX", "continent": 2, "country": "SWEDEN"},
           {"ticker": "PSI20.LS", "continent": 2, "country": "PORTUGAL"},
           {"ticker": "^GSPTSE", "continent": 1, "country": "CANADA"},
           {"ticker": "^SPBLPGPT", "continent": 0, "country": "PERU"},
           {"ticker": "^NZ50", "continent": 0, "country": "NEW ZEALAND"},
           {"ticker": "CIH", "continent": 3, "country": "CHINA"},
           {"ticker": "KEMX", "continent": 3, "country": "CHINA"},
           {"ticker": "^KLSE", "continent": 3, "country": "Malaysia"},
           {"ticker": "^TECDAX", "continent": 2, "country": "GERMANY"},
           {"ticker": "^STOXX50E", "continent": 2, "country": "EU"},
           {"ticker": "^NYA", "continent": 1, "country": "USA"},
           {"ticker": "^RUT", "continent": 3, "country": "RUSSIA"},
           {"ticker": "^JKSE", "continent": 3, "country": "INDONESIA"},
           {"ticker": "^N100", "continent": 2, "country": "EU"},
           {"ticker": "^BFX", "continent": 2, "country": "BELGUIM"},
           {"ticker": "^SPGTAQUA", "continent": 0, "country": "ROW"},
           {"ticker": "AW01.FGI", "continent": 0, "country": "ROW"},
           {"ticker": "^W5000", "continent": 1, "country": "US"}
           ]
"""
path = "data/index_{}".format(measurement)
if add_range:
    path = "{}_range_{}mo".format(path, str(m_intervall))
else:
    path = "{}_{}mo".format(path, str(m_intervall))
if log:
    path = "{}_log".format(path)
if normalization:
    path = "{}_{}_norm".format(path, normalize_growth)
if not os.path.exists(path):
    os.makedirs(path)
f = open("{}/description.txt".format(path), "a")
f.write("DESCRIPTION OF THE FINANCE DATASET\n")
f.write("Measurement: \t{}\n".format(measurement))
f.write("Start year: \t{}\n".format(start_year))
f.write("End year: \t{}\n".format(end_year))
f.write("Max Len: {}\n".format(str(max_seq_len)))
f.write("Data Attribute:\n")
f.write("continent \t= \t[0, 1, 2, 3]    # [Rest of the World, North America, Europe, Asia])\n")
f.write("decade \t= \t[0, 1, 2]  # [90's, 00's, 10's]\n")

for index in indices:
    index_counter = 0
    # iterate over all dates
    for year in range(start_year, end_year, 1):
        for m_idx in range(0, len(months), m_intervall):
            start_date = str(year) + "-" + months[m_idx] + "-01"
            if m_idx >= (len(months) - m_intervall):
                end_date = str(year + 1) + "-" + months[0] + "-01"
            else:
                end_date = str(year) + "-" + months[m_idx + m_intervall] + "-01"

            raw_data = yf.download(tickers=index['ticker'],
                                   start=start_date,
                                   end=end_date,
                                   interval="1d",
                                   group_by='ticker',
                                   auto_adjust=True,
                                   treads=True)
            if len(raw_data) < (max_seq_len / 2):
                continue
            raw_data['Open'] = raw_data['Open'].fillna(method='ffill')
            raw_data['Close'] = raw_data['Close'].fillna(method='ffill')
            raw_data['High'] = raw_data['High'].fillna(method='ffill')
            raw_data['Low'] = raw_data['Low'].fillna(method='ffill')
            raw_data['Growth'] = raw_data['Close'] / raw_data['Open']
            # check for equality
            if raw_data['Open'].equals(raw_data['Close']):
                continue
            if log:
                raw_data['Growth'] = np.log(raw_data['Growth'])
            # https://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame

            if normalize_growth == "mean":
                raw_data['Growth_norm'] = (raw_data['Growth'] - raw_data['Growth'].mean()) / raw_data['Growth'].std()
            elif normalize_growth == 'minusone_one':
                raw_data['Growth_norm'] = 2 * ((raw_data['Growth'] - raw_data['Growth'].min()) / (
                        raw_data['Growth'].max() - raw_data['Growth'].min())) - 1
            elif normalize_growth == 'zero_one':
                raw_data['Growth_norm'] = (raw_data['Growth'] - raw_data['Growth'].min()) / (
                        raw_data['Growth'].max() - raw_data['Growth'].min())
            else:
                raw_data['Growth_norm'] = (raw_data['Growth'] - raw_data['Growth'].min()) / (
                        raw_data['Growth'].max() - raw_data['Growth'].min())

            raw_data['Range'] = raw_data['High'] - raw_data['Low']

            if normalize_range == "mean":
                raw_data['Range_norm'] = (raw_data['Range'] - raw_data['Range'].mean()) / raw_data['Range'].std()
            elif normalize_range == 'minusone_one':
                raw_data['Range_norm'] = 2 * ((raw_data['Range'] - raw_data['Range'].min()) / (
                        raw_data['Range'].max() - raw_data['Range'].min())) - 1
            elif normalize_range == 'zero_one':
                raw_data['Range_norm'] = (raw_data['Range'] - raw_data['Range'].min()) / (
                        raw_data['Range'].max() - raw_data['Range'].min())
            else:
                raw_data['Range_norm'] = (raw_data['Range'] - raw_data['Range'].min()) / (
                        raw_data['Range'].max() - raw_data['Range'].min())
            if raw_data['Growth_norm'].isnull().values.any():
                continue
            if add_range and raw_data['Range_norm'].isnull().values.any():  #
                continue
            # add attributes
            # continent
            cont = np.zeros(len(continent))
            cont[index["continent"]] = 1
            # decade
            dec = np.zeros(len(decade))
            dec[calc_decade(year)] = 1
            attribute = np.expand_dims(np.concatenate((cont, dec)), axis=0)
            data_attribute = np.concatenate((data_attribute, attribute), axis=0)
            # add features
            # price or growth
            feature = np.zeros((max_seq_len, feat_range))
            if measurement == "price":
                current_course = np.asarray(raw_data['Open'])
            else:
                if normalization:
                    current_course = np.asarray(raw_data['Growth_norm'])
                    current_range = np.asarray(raw_data['Range_norm'])
                else:
                    current_course = np.asarray(raw_data['Growth'])
                    current_range = np.asarray(raw_data['Range'])
            feature[:len(current_course), 0] = current_course
            if add_range:
                feature[:len(current_range), 1] = current_range
            feature = np.expand_dims(feature, axis=0)
            data_feature = np.concatenate((data_feature, feature), axis=0)
            # add gen flag
            gen_flag = np.zeros(max_seq_len)
            gen_flag[:len(current_course)] = 1
            data_gen_flag = np.concatenate((data_gen_flag, np.expand_dims(gen_flag, axis=0)), axis=0)
            index_counter += 1
    index['# samples'] = index_counter
    print("{} finished - current len: {}".format(index['ticker'], len(data_attribute)))

f.write("INDICES\n")
f.write(str(pd.DataFrame.from_dict(indices)))
f.write("\nTotal length: {}\n".format(str(len(data_attribute))))
attr_counter = 0
for cont in continent:
    nr_data_points = np.count_nonzero(data_attribute[:, attr_counter])
    f.write("{} nr data points: {}\n".format(cont, str(nr_data_points)))
    attr_counter += 1
for dec in decade:
    nr_data_points = np.count_nonzero(data_attribute[:, attr_counter])
    f.write("{} nr data points: {}\n".format(dec, str(nr_data_points)))
    attr_counter += 1
f.close()
dbfile = open('{}/data_attribute_output.pkl'.format(path), 'ab')
pickle.dump(data_attribute_outputs, dbfile)
dbfile.close()
dbfile = open('{}/data_feature_output.pkl'.format(path), 'ab')
pickle.dump(data_feature_outputs, dbfile)
dbfile.close()

np.savez("{}/data_train.npz".format(path), data_attribute=data_attribute, data_feature=data_feature,
         data_gen_flag=data_gen_flag)
