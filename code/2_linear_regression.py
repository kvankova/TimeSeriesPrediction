import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats

from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import _functions as qtf

plt.rcParams['figure.figsize'] = [10, 8]

#  Load data
data, train_data, test_data = qtf.load_data()

data = data.dropna()
train_data = train_data.dropna()
test_data = test_data.dropna()

# Plot log-transformed time series of train data
plt.plot(data['Date'], data['log_volume'])
plt.xlabel('Date')
plt.ylabel('log(volume)')
plt.savefig('all_data_log_volume.png')
plt.show()

# Additive Decomposition
decomposition = seasonal_decompose(train_data.set_index('Date')['log_volume'], model='additive', period=250)
decomposition.plot()
plt.savefig('additive_decomposition.png')


# Plot boxplots of log-transformed volume by Quarter, Month, Week
fig, ((ax1, ay1)) = plt.subplots(1, 2, figsize=(20,10))
train_data.boxplot(column='log_volume', by='Quarter', ax=ax1)
train_data.boxplot(column='log_volume', by='Month', ax=ay1)
fig.suptitle('Boxplots of log-transformed volume by Quarter and Month')
plt.savefig('train_data_log_volume_boxplot.png')

# Linear regression
window = len(train_data)

ohe_columns = ['Month', 'DayOfWeek', 'Day']
numeric_columns = ['time','time2', 'time3', 'log_time', 'prev1_volume','prev2_volume', 'prev3_volume', 'prev4_volume']

test_data['log_volume_preds'] = np.nan
test_data['volume_preds'] = np.nan
stdev_train = []

for pred_step in test_data.index:
    print(pred_step)

    train_data_window = data.loc[pred_step - window : pred_step - 1]
    y_train_window = train_data_window['log_volume']

    test_data_one_row = test_data.loc[[pred_step]]
    y_test_one_row = test_data['log_volume'].loc[pred_step]
   
    preprocessor = ColumnTransformer([
        ('numeric', StandardScaler(), numeric_columns),
        ('ohe', OneHotEncoder(), ohe_columns)   
    ])

    pipeline = Pipeline([('preprocessor', preprocessor), 
                         ('lr', LinearRegression())])
    
    pipeline.fit(train_data_window, y_train_window)

    # calculate stdev of train residuals for confidence interval of predictions
    stdev_train.append(np.sqrt(sum((np.exp(pipeline.predict(train_data_window)) - np.exp(y_train_window))**2) / (len(y_train_window) - 2)))

    test_log_pred = pipeline.predict(test_data_one_row)

    test_data['log_volume_preds'].loc[pred_step] = test_log_pred
    test_data['volume_preds'].loc[pred_step] = np.exp(test_log_pred)


date = test_data['Date']
preds = test_data['volume_preds']
actual = test_data['Volume']

# Plot predictions vs actual data
qtf.plot_preds_actual(date, preds, actual, 'Linear Regression vs test data', 'linear_regression_prediction_interval',stdev_train)

# Calculate R2, SSE, residuals
sse, r2, residuals = qtf.calculate_metrics(preds = preds.astype(int), actual = actual)

residuals.describe()

# Plot residuals diagnostics plot
qtf.plot_residuals_diagnostics(residuals, preds, date, 'Linear Regression')


# Get coefficients for the first fit using statsmodels
# get feature names from scaler
scaler = (pipeline.named_steps['preprocessor'].named_transformers_['numeric'])
scaler_feature_names = list(scaler.get_feature_names_out(numeric_columns))

# get feature names from onehot
ohe = (pipeline.named_steps['preprocessor'].named_transformers_['ohe'])
ohe_feature_names = list(ohe.get_feature_names_out(input_features=ohe_columns))

# get coefficients for the first fit
train_data_window_transformed = pipeline[0].fit_transform(train_data_window).toarray()
X_pd = pd.DataFrame(train_data_window_transformed, columns=scaler_feature_names + ohe_feature_names)
X = sm.add_constant(train_data_window_transformed)  

model = sm.OLS(y_train_window, X).fit()
model.summary(xname=['const'] + scaler_feature_names + ohe_feature_names)