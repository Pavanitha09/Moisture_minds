from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from tqdm import tqdm_notebook
from itertools import product
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import sys
import pickle

import warnings
warnings.filterwarnings('ignore')



# custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
df = pd.read_csv(sys.argv[1])
# df = pd.read_csv('/content/trail_file.csv')
swap_list = ['pm1', 'pm2', 'pm3', 'am', 'lum', 'st','pres','humd','temp','sm']

# Swapping the columns
df = df.reindex(columns=swap_list)
# print(df.head())

cols = list(df)[0:10]
df_for_training = df[cols].astype(float)
scaler = StandardScaler()
scaler = scaler.fit(df_for_training)
df_test = scaler.transform(df_for_training)

testX = []
testY = []

n_future = 1   # Number of days we want to look into the future based on the past days.
n_past = 14 # Number of past days we want to use to predict the future.
for i in range(n_past, len(df_test) - n_future +1):
    testX.append(df_test[i - n_past:i, 0:df_test.shape[1]]-1)
    testY.append(df_test[i + n_future - 1:i + n_future, -1])

#loading model
model_file = open('model.pkl', 'rb')     
model = pickle.load(model_file)

# trainX, trainY = np.array(trainX), np.array(trainY)
testX = np.array(testX)
testY = np.array(testY)
print(testX.shape)
y_pred = model.predict(testX)
prediction_copies = np.repeat(y_pred, df_for_training.shape[1], axis=-1)
real_predictions = scaler.inverse_transform(prediction_copies)[:,-1]
y_copies = np.repeat(testY, df_for_training.shape[1], axis=-1)
y = scaler.inverse_transform(y_copies)[:,-1]
# print(y)
# print(real_predictions)

# RMSE computation

# rmse = np.sqrt(np.mean(((real_predictions - y) ** 2)))
# print('Root Mean Squared Error:', rmse)

# Writing to csv file

pred_values = pd.DataFrame(real_predictions)
pred_values.to_csv('results.csv')
