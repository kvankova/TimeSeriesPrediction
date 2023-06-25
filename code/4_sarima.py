import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.stattools import adfuller
import pandas as pd
from itertools import product
from statsmodels.tsa.statespace.sarimax import SARIMAX

import _functions as qtf

plt.rcParams['figure.figsize'] = [10, 8]

#  Load data
data, train_data, test_data = qtf.load_data()

# difference log of volume data
data['log_volume_diff'] = data['log_volume'].diff()

# Plot difference 
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,10))
ax1.plot(data['Date'], data['Volume'].diff())
ax1.set_ylabel('Difference')
ax1.set_xlabel('Date')
ax1.set_title('1st difference of Volume')

ax2.plot(data['Date'], data['log_volume_diff'])
ax2.set_ylabel('Difference')
ax2.set_xlabel('Date')
ax2.set_title('1st difference of log(Volume)')

plt.savefig('sarima_diff_volume.png')

data = data.dropna()

# seasonal difference
data['log_volume_diff_seasonal'] = data['log_volume_diff'].diff(125)

# plot acf and pacf of 1st difference
fig, ((ax1, ay1), (ax2, ay2)) = plt.subplots(2, 2, figsize=(20,20))
plot_acf(data['log_volume_diff'], lags=50, ax=ax1) 
ax1.set_title('ACF of 1st difference of log(volume)')
plot_pacf(data['log_volume_diff'], lags=50, ax=ay1) 
ay1.set_title('PACF of 1st difference of log(volume)')

# plot acf and pacf of seasonal difference
plot_acf(data['log_volume_diff_seasonal'].dropna(), lags=150, ax=ax2) 
ax2.set_title('ACF of 1st difference of log(volume), and seasonal difference of 125') 
plot_pacf(data['log_volume_diff_seasonal'].dropna(), lags=150, ax=ay2) 
ay2.set_title('PACF of 1st difference of log(volume), and seasonal difference of 125')

plt.savefig('sarima_acf_pacf.png')

# test for stationarity
ad_fuller_result = adfuller(data['log_volume_diff'])
print(f'ADF Statistic: {ad_fuller_result[0]}')
print(f'p-value: {ad_fuller_result[1]}') # 0 < 0.05, reject null hypothesis, data is stationary

# fit SARIMA for all parameters
p = range(0, 4, 1)
d = 1
q = range(0, 4, 1)
P = range(0, 4, 1)
D = 1
Q = range(0, 4, 1)
s = 125
parameters = product(p, q, P, Q)
parameters_list = list(parameters)
print(len(parameters_list))

sarima_result_df = qtf.fit_sarima(parameters_list, d, D, s, data['log_volume_diff_seasonal'])
print('Best SARIMA model: ', sarima_result_df.iloc[0])

# fit ARIMA for all parameters
p = range(0, 4, 1)
d = 1
q = range(0, 4, 1)
P = range(0, 1, 1)
D = 0
Q = range(0, 1, 1)
s = 250
parameters = product(p, q, P, Q)
parameters_list = list(parameters)
print(len(parameters_list))
arima_result_df = qtf.fit_sarima(parameters_list, d, D, s, data['log_volume_diff'])
print('Best ARIMA model: ', arima_result_df.iloc[0])

# Perform one-day ahead predictions with the final model
window = len(train_data)
test_data['log_volume_preds'] = np.nan
test_predictions = []
stdev_train = []

for pred_step in test_data.index:
    print(pred_step)
    
    train_data_window = data.loc[pred_step - window : pred_step - 1]

    arima_model = SARIMAX(train_data_window['log_volume'], order=(3, 1, 2), seasonal_order=(0, 0, 0, 0)).fit(disp=-1)
    
    # calculate stdev of train residuals for confidence interval of predictions
    stdev_train.append(np.sqrt(sum((np.exp(arima_model.fittedvalues) - train_data_window['Volume'])**2) / (len(train_data_window) - 2)))
    
    # make one-day ahead prediction
    test_predictions.append(arima_model.forecast(1).values[0])

test_data['volume_preds'] = np.exp(test_predictions)

# Plot residuals
date = test_data['Date']
preds = test_data['volume_preds']
actual = test_data['Volume']

# Plot predictions vs actual data
qtf.plot_preds_actual(date, preds, actual, 'ARIMA vs test data', 'arima_prediction_interval', stdev_train)

sse, r2, residuals = qtf.calculate_metrics(test_data['volume_preds'], test_data['Volume'])

residuals.describe()

qtf.plot_residuals_diagnostics(residuals, preds, date, 'ARIMA')


