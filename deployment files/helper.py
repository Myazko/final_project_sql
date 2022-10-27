import pandas as pd
import pickle
from columns import model_columns
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import numpy as np
from inputs import payload
#from scalers import scaler, scaler_pred
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
# from inputs import payload
from columns import model_columns
from inputs import payload
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# from sklearn.metrics import mean_absolute_error, mean_squared_error # Packages for measuring model performance / errors
# from tensorflow.keras.models import Sequential # Deep learning library, used for neural networks
# from tensorflow.keras.layers import LSTM, Dense, Dropout # Deep learning classes for recurrent and regular densely-connected layers
# from tensorflow.keras.callbacks import EarlyStopping # EarlyStopping during model training
# from sklearn.preprocessing import RobustScaler, MinMaxScaler # This Scaler removes the median and scales the data according to the quantile range to normalize the price data
# import seaborn as sns # Visualization
from tensorflow import keras

#from scalers import scaler,scaler_pred
#from scaleconverter import scale_converter

from sklearn.preprocessing import RobustScaler, MinMaxScaler
Sclaear=MinMaxScaler()

import datetime as dt
import pandas as pd
import numpy as np


from datetime import datetime, timedelta

from pathlib import Path

import re

from sklearn.preprocessing import StandardScaler

# Time Series Forecasting - Multivariate Time Series Models for Stock Market Prediction
# A tutorial for this file is available at www.relataly.com
import math # Mathematical functions 
import numpy as np # Fundamental package for scientific computing with Python
import pandas as pd # Additional functions for analysing and manipulating data
from datetime import date, timedelta, datetime # Date Functions
from pandas.plotting import register_matplotlib_converters # This function adds plotting functions for calender dates
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error # Packages for measuring model performance / errors
from tensorflow.keras.models import Sequential # Deep learning library, used for neural networks
from tensorflow.keras.layers import LSTM, Dense, Dropout # Deep learning classes for recurrent and regular densely-connected layers
from tensorflow.keras.callbacks import EarlyStopping # EarlyStopping during model training
from sklearn.preprocessing import RobustScaler, MinMaxScaler # This Scaler removes the median and scales the data according to the quantile range to normalize the price data 









def data_preprocessing(payload):
    df =pd.DataFrame.from_dict(payload, orient= 'index').T
    # x = pd.get_dummies(df)
    # print(df.info()) #
    # let's convert objects to valid types
    df['Open'] = df['Open'].astype(float)
    df['High'] = df['High'].astype(float)
    df['Low'] = df['Low'].astype(float)
    df['Close'] = df['Close'].astype(float)
    df['Adj Close'] = df['Adj Close'].astype(float)
    df['Volume'] = df['Volume'].astype(int)
    df['Confirmed'] = df['Confirmed'].astype(float)
    df['Death'] = df['Death'].astype(float)
    df['Recovered'] = df['Recovered'].astype(float)
    df['sentiment'] = df['sentiment'].astype(float)

    x = pd.get_dummies(df)

    for column in model_columns:
        if column not in x.columns:
            x[column] = 0
    x = x[model_columns]
    print (x)

    return x




dateparse = lambda dates: datetime.strptime(dates[:10], '%Y-%m-%d')

Financial_Data_Covid_Data_Tweet_Data= pd.read_csv('all_data_step_4_copy.csv',  index_col='Date',
                 parse_dates = ['Date'], date_parser = dateparse
                )
train_df = Financial_Data_Covid_Data_Tweet_Data.sort_values(by=['Date']).copy()

# List of considered Features
FEATURES = ['Open','High', 'Low', 'Close', 'Adj Close', 'Volume', 'Confirmed', 'Death', 'Recovered', 'sentiment'
            #, 'Month', 'Year', 'Adj Close'
            # I ran the code with the dates to investigate the impact of the features on different dates. I will do that in a future update
           ]

print('FEATURE LIST')
print([f for f in FEATURES])

# Create the dataset with features and filter the data to the list of FEATURES
data = pd.DataFrame(train_df)
data_filtered = data[FEATURES]

# We add a prediction column and set dummy values to prepare the data for scaling
data_filtered_ext = data_filtered.copy()
data_filtered_ext['Prediction'] = data_filtered_ext['Adj Close']

# Print the tail of the dataframe

# Get the number of rows in the data
nrows = data_filtered.shape[0]

# Convert the data to numpy values
np_data_unscaled = np.array(data_filtered)
np_data = np.reshape(np_data_unscaled, (nrows, -1))
print(np_data.shape)

# Transform the data by scaling each feature to a range between 0 and 1
scaler = MinMaxScaler()
np_data_scaled = scaler.fit_transform(np_data_unscaled)

# Creating a separate scaler that works on a single column for scaling predictions
scaler_pred = MinMaxScaler()
df_Close = pd.DataFrame(data_filtered_ext['Adj Close'])
np_Close_scaled = scaler_pred.fit_transform(df_Close)
np_data_scaled





def prediction(payload, filename):
    last_N_days=data_preprocessing(payload)
    last_N_days = last_N_days.values
    last_N_days_scaled = scaler.transform(last_N_days)
    X_test_new = []
    X_test_new.append(last_N_days_scaled)
    model = tf.keras.models.load_model(filename)
    pred_price_scaled = model.predict(np.array(X_test_new))
    pred_price_unscaled = scaler_pred.inverse_transform(pred_price_scaled.reshape(-1, 1))
    print(pred_price_unscaled)
    return pred_price_unscaled


