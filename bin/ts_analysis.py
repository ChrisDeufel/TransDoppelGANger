import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as sgt
import statsmodels.tsa.stattools as sts
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
sns.set()

dax_path = "data/finance/csvs/DAX.csv"
raw_df = pd.read_csv(dax_path, delimiter=';')
df = raw_df.copy()
del df['High']
del df['Low']
df.Date = pd.to_datetime(df.Date, dayfirst=True)
df.set_index("Date", inplace=True)
#df = df.asfreq('b')
df['Open'] = df['Open'].str.replace(".", "").astype(float)
df = df.fillna(method='ffill')
sgt.plot_acf(df.Open, lags=40)
plt.title("DAX", size=24)
plt.show()

print('hello')

