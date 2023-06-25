import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
import numpy as np

import _functions as qtf

plt.rcParams['figure.figsize'] = [10, 8]

#  Load data
data, train_data, test_data = qtf.load_data()

# Set date as index
data['old_index'] = data.index
train_data['old_index'] = train_data.index
test_data['old_index'] = test_data.index

data=data.set_index('Date')
train_data=train_data.set_index('Date')
test_data=test_data.set_index('Date')

# Set alpha and time period
m = 250
alpha = 1/(2*m)

# Exploring simple, double and triple smoothing
## simple exponential smoothing, no trend, no seasonality, evaluate fit
ses = SimpleExpSmoothing(train_data['Volume'])
ses_fit = ses.fit(smoothing_level=alpha)
print(ses_fit.summary())

train_data['pred_SES'] = ses_fit.fittedvalues
train_data[['Volume','pred_SES']].plot(title='Holt Winters Single Exponential Smoothing')

## Double exponential smoothing, with trend, no seasonality
train_data['pred_DES_add'] = ExponentialSmoothing(train_data['Volume'],trend='add').fit().fittedvalues
train_data['pred_DES_mul'] = ExponentialSmoothing(train_data['Volume'],trend='mul').fit().fittedvalues
train_data[['Volume','pred_DES_add', 'pred_DES_mul']].plot(title='Holt Winters Double Exponential Smoothing: Additive and Multiplicative Trend')

_, r2, _ = qtf.calculate_metrics(train_data['pred_DES_mul'], train_data['Volume'])

## Triple exponential smoothing, with trend, with seasonality
fitted_model = ExponentialSmoothing(train_data['Volume'], trend = 'add' , seasonal='add', seasonal_periods=250).fit()
                

print(fitted_model.summary())
test_predictions = fitted_model.forecast(len(test_data))
test_predictions.index = test_data.index

### Plot predictions vs actual data
test_data['Volume'].plot(legend=True,label='TEST',figsize=(12,8))
test_predictions.plot(legend=True,label='PREDICTION')
plt.title('Test and predicted test data using Holt Winters Triple Exponential Smoothing')

sse, r2, residuals = qtf.calculate_metrics(test_predictions, test_data['Volume'])

# Perform one-day ahead predictions with the final model
window = len(train_data)
test_data['volume_preds'] = np.nan
test_predictions = []
stdev_train = []

for i, pred_step in enumerate(list(test_data['old_index'])):
    print(pred_step)
    
    train_data_window = data.loc[(data['old_index'] <= pred_step - 1) & (data['old_index'] >= pred_step - window), 'Volume']

    fitted_model = (
        ExponentialSmoothing(train_data_window, 
                             trend = 'add', 
                             seasonal='add', 
                             seasonal_periods=250
                             )
                             .fit()
                             )
    
    # calculate stdev of train residuals for confidence interval of predictions
    stdev_train.append(np.sqrt(sum((fitted_model.fittedvalues - train_data_window)**2) / (len(train_data_window) - 2)))
    
    # make one-day ahead prediction
    test_predictions.append(fitted_model.forecast(1).values[0])

test_data['volume_preds'] = test_predictions

# Plot residuals
date = test_data.index
preds = test_data['volume_preds']
actual = test_data['Volume']

# Plot predictions vs actual data
qtf.plot_preds_actual(date, preds, actual, 'Holt Winters method vs test data', 'holt_winters_prediction_interval', stdev_train)

sse, r2, residuals = qtf.calculate_metrics(test_data['volume_preds'], test_data['Volume'])

residuals.describe()

qtf.plot_residuals_diagnostics(residuals, preds, date, 'Holt Winters')
