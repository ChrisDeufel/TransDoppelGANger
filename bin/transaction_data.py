from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
import pandas as pd
import tqdm
import numpy as np
import pickle
from output import Output, OutputType, Normalization

########### prepare transaction dataset for DoppelGANger framework ###########
# utils
def timeEncoder(X):
    X_hm = X['Time'].str.split(':', expand=True)
    hour = X_hm[0].astype(int)
    minute = X_hm[1].astype(int)
    d = pd.to_datetime(dict(year=X['Year'], month=X['Month'], day=X['Day'], hour=hour, minute=minute))
    return pd.DataFrame(d)

def label_fit_transform(column, enc_type="label"):
    if enc_type == "label":
        mfit = OneHotEncoder()
    else:
        mfit = MinMaxScaler()
    mfit.fit(column)
    return mfit, mfit.transform(column)

def amountEncoder(X):
    amt = X.apply(lambda x: x[1:]).astype(float).apply(lambda amt: max(1, amt)).apply(math.log)
    return pd.DataFrame(amt)

def fraudEncoder(X):
    fraud = (X == 'Yes').astype(int)
    return pd.DataFrame(fraud)

def nanNone(X):
    return X.where(pd.notnull(X), 'None')

def nanZero(X):
    return X.where(pd.notnull(X), 0)

# read dataset
data_path = 'data/transactions/card_transaction.v1.csv'
data = pd.read_csv(data_path)
max_len = 50000

# instantiate meta data
data_attribute_outputs = []
data_feature_outputs = []
# initiate numpy arrays for data attributes and features
data_feature = data['User'].to_numpy()
data_feature = np.reshape(data_feature, (data_feature.shape[0], 1))
data_attribute = data_feature
# add to meta data and write to text file
data_attribute_outputs.append(Output(type_=OutputType.DISCRETE, dim=1))

des = open("transactions_description.txt", "w+")
des.write("IMPORTANT NOTES - \n"
          "MAX LEN PER SAMPLE IS LIMITED TO {0}\n"
          "MANY COLUMNS (ZIP, MERCHANT STATE ETC.) ARE MISSING DUE TO MEMORY ALLOCATION PROBLEMS".format(max_len))
des.write("Attribute - Column 0: User\n")

# prepare columns
#data['Errors?'] = nanNone(data['Errors?'])
data['Is Fraud?'] = fraudEncoder(data['Is Fraud?'])
data['Zip'] = nanZero(data['Zip'])
data['Merchant State'] = nanNone(data['Merchant State'])
data['Use Chip'] = nanNone(data['Use Chip'])

# columns with continuous features
data['Amount'] = data['Amount'].str.replace('$', '')
amount = data['Amount'].to_numpy().astype(np.float)
amount = np.reshape(amount, (amount.shape[0], 1))
data_feature = np.concatenate((data_feature, amount), axis=1)
# add to meta data and write to text file
data_feature_outputs.append(Output(type_=OutputType.CONTINUOUS, dim=1, normalization=Normalization.MINUSONE_ONE))
des.write("Feature - Column 0: Amount\n")
amount = None

timestamp = timeEncoder(data[['Year', 'Month', 'Day', 'Time']])
timestamp_fit, timestamp = label_fit_transform(timestamp, enc_type="time")
data_feature = np.concatenate((data_feature, timestamp), axis=1)
# add to meta data and write to text file
data_feature_outputs.append(Output(type_=OutputType.CONTINUOUS, dim=1, normalization=Normalization.ZERO_ONE))
des.write("Feature - Column 1: Timestamp\n")
timestamp = None

# columns with categorical features
sub_columns = ['Use Chip', 'Card']
col_len = data_feature.shape[1] - 1

for col_name in tqdm.tqdm(sub_columns):
    col_data = data[col_name].to_numpy()
    col_data = np.reshape(col_data, (col_data.shape[0], 1))
    col_fit, col_data = label_fit_transform(col_data)
    data_feature = np.concatenate((data_feature, col_data.toarray()), axis=1)
    # add to meta data and write to text file
    data_feature_outputs.append(Output(type_=OutputType.DISCRETE, dim=col_data.shape[1]))
    des.write("Feature - Column {0}: {1}\n".format(col_len+1, col_name))
    col_len += col_data.shape[1]
data = None
col_data = None

# save numpy arrays and output objects
dbfile = open('data/transactions/data_attribute_output.pkl', 'ab')
# source, destination
pickle.dump(data_attribute_outputs, dbfile)
dbfile.close()
# source, destination
dbfile = open('data/transactions/data_feature_output.pkl', 'ab')
pickle.dump(data_feature_outputs, dbfile)
dbfile.close()
# determine max len of transactions for one user
user = data_feature[:, 0].astype(int)
user_count = np.bincount(user)
nr_user = len(user_count)
max_user = np.argmax(user_count)
#max_len = user_count[max_user]


# take user column of data feature
data_feature = data_feature[:, 1:]

# instantiate and fill final arrays
data_gen_flags = np.zeros((nr_user, max_len))
#data_attribute_final = np.reshape(np.arange(nr_user), (nr_user, 1))
#data_feature_final = np.zeros((nr_user, max_len, data_feature.shape[1]))
#data_gen_flag = np.zeros((nr_user, max_len))
current = 0
for u in range(nr_user):
    #feature = np.zeros((max_len, data_feature.shape[1]))
    seq_len = max_len
    if user_count[u] < seq_len:
        seq_len = user_count[u]
    data_gen_flags[u, :seq_len] = 1
    data_u = data_feature[current:current + seq_len]
    #feature[:seq_len, :] = data_u
    #data_feature_final[u, :seq_len, :] = np.reshape(data_u, (1, seq_len, data_feature.shape[1]))
    #data_gen_flag[u, :seq_len] = 1
    #np.save("data/transactions/{}_data_feature.npy".format(u), feature)
    current += user_count[u]
np.save("data/transactions/data_gen_flag.npy", data_gen_flags)



#np.save('transactions_features.npy', data_feature_final)
#np.save('transactions_attributes.npy', data_attribute_final)
#np.savez('data_train.npz', data_feature=data_feature_final, data_attribute=data_attribute_final,
#         data_gen_flag=data_gen_flag)





