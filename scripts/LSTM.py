import pandas as pd
import numpy as np

import keras


data_c = pd.read_csv("data/final/1min/BTC.csv")
data_c.drop("system_time", axis = 1)
data_c['Next_mid'] = data_c.midpoint.shift(-1)
data_c['Price_move'] = (data_c.Next_mid - data_c.midpoint) / data_c.midpoint
data_c.dropna(axis=0, inplace=True) #drop the last row which is now Nan

data_c['y'] = 0
data_c.loc[(data_c['Price_move'] < -0.0005), 'y'] = 1
data_c.loc[(data_c['Price_move'] > 0.0005), 'y'] = 2
data_c.drop(['Next_mid', 'midpoint', 'Price_move', "system_time", "Return"], axis=1, inplace=True)


Y = data_c.pop("y")


bids = data_c.iloc[:,0:18]
asks = data_c.iloc[:,78:93]
data_bid_ask  = pd.concat([bids.reset_index(drop=True), asks], axis=1)


