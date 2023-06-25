import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, acf, pacf
from statsmodels.tsa.stattools import adfuller

import _functions as qtf

plt.rcParams['figure.figsize'] = [10, 8]

#  Load data
data, train_data, test_data = qtf.load_data()

# Analyze data
data['Volume'].describe()
data['Volume'].isna().sum()

# get skewness and curtosis of data
print('Skewness: ', data['Volume'].skew())
print('Kurtosis: ', data['Volume'].kurtosis())

# ACF, PACF of train data
fig, ((ax1, ay1)) = plt.subplots(1, 2, figsize=(20,10))
plot_acf(train_data['Volume'], ax =ax1)
plot_pacf(train_data['Volume'], ax=ay1)
fig.suptitle('Volume autocorrelation and partial autocorrelation, train data')
plt.savefig('train_data_acf_pacf.png')

print('ACF(1), ACF(20), ACF(250): ', acf(train_data['Volume'], nlags=251)[[1, 20, 250]])
print('PACF(1), PACF(20), PACF(250): ', pacf(train_data['Volume'], nlags=251)[[1, 20, 250]])
 
 # Test stationarity hypothesis with Augmented Dickey-Fuller test
print('Augmented Dickey-Fuller test: ', adfuller(train_data['Volume']))

fig, ((ax1, ay1)) = plt.subplots(1, 2, figsize=(20,8))

# Plot time series of all data
ax1.plot(data['Date'], data['Volume'])
ax1.set_title('Time series of S&P stock index volume')
ax1.set_xlabel('Date')
ax1.set_ylabel('Volume')

# Plot histogram of train data
train_data['Volume'].hist(bins = 50, ax=ay1)
ay1.set_title('Histogram of S&P stock index volume')
ay1.set_xlabel('Volume')
ay1.set_ylabel('Frequency')

plt.savefig('all_data_volume_ts_hist.png')
