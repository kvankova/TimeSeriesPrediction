import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
from darts.utils.likelihood_models import QuantileRegression
from darts.utils.timeseries_generation import datetime_attribute_timeseries

import _functions as qtf


# quantiles for QuantileRegression
QUANTILES = [0.01, 0.05, 0.1, 0.2, 0.25, 0.5, 0.75, 0.8, 0.9, 0.95, 0.99]

#  Load data
data, train_data, test_data = qtf.load_data()
training_cutoff = pd.Timestamp('2017-01-01')
series = TimeSeries.from_dataframe(data, "Date", "Volume", freq='B', fillna_value=0)

train_series, test_series = series.split_after(pd.Timestamp(training_cutoff))

# Scale data
transformer = Scaler()
train_series_scaled = transformer.fit_transform(train_series)
test_series_scaled = transformer.transform(test_series)
series_scaled = transformer.transform(series)

# create covariates: year, quarter, month,... and integer index series
cov = datetime_attribute_timeseries(series, attribute="year", one_hot=False)
cov = cov.stack(datetime_attribute_timeseries(series, attribute="quarter", one_hot=False))
cov = cov.stack(datetime_attribute_timeseries(series, attribute="month", one_hot=False))
cov = cov.stack(datetime_attribute_timeseries(series, attribute="week", one_hot=False))
cov = cov.stack(datetime_attribute_timeseries(series, attribute="day", one_hot=False))
cov = cov.stack(datetime_attribute_timeseries(series, attribute="dayofweek", one_hot=False))
cov = cov.stack(TimeSeries.from_times_and_values(
                                    times=series.time_index, 
                                    values=np.arange(len(series)), 
                                    columns=["linear_increase"]))
cov = cov.astype(np.float32)

# train/test
train_cov, test_cov = cov.split_after(training_cutoff)

# rescale the covariates: fit on the training set
scaler = Scaler()
scaler.fit(train_cov)
tcov = scaler.transform(cov)

# Instantiate model
model = TFTModel(input_chunk_length=32,
                    output_chunk_length=8,
                    hidden_size=64,
                    hidden_continuous_size=64,
                    full_attention=True, 
                    lstm_layers=1,
                    num_attention_heads=3,
                    dropout=0,
                    batch_size=16,
                    n_epochs=30,
                    likelihood=QuantileRegression(quantiles=QUANTILES), 
                    random_state=42, 
                    force_reset=True)


model.fit(  train_series_scaled, 
            future_covariates=tcov, 
            verbose=True)   


# Make predictions
ts_tpred = model.predict(   n=len(test_series_scaled), 
                            num_samples=100,   
                            n_jobs=-1)
    
# Inverse scaler transformation
ts_pred = transformer.inverse_transform(ts_tpred)

# Calculate mean prediction and filter business days
pd_pred = ts_pred.pd_dataframe()
business_days = pd_pred.index.isin(test_data['Date'])
pd_pred_business = pd_pred[business_days].copy()

pd_pred_mean = pd_pred_business.mean(axis=1)
pd_pred_std = pd_pred_business.std(axis=1)

pd_pred_mean_stepidx = pd_pred_mean.reset_index().set_index(test_data.index)[0]
pd_pred_std_stepidx = pd_pred_std.reset_index().set_index(test_data.index)[0]


# Plot residuals
date = test_data['Date']
preds = pd_pred_mean_stepidx
actual = test_data['Volume']
quantile_pred_std = pd_pred_std_stepidx

# Plot predictions vs actual data
qtf.plot_preds_actual(date, preds, actual, 'TFT model vs test data', 'tft_prediction_interval', quantile_pred_std)

# Calculate metrics
sse, r2, residuals = qtf.calculate_metrics(pd_pred_mean_stepidx, test_data['Volume'])

residuals.describe()

qtf.plot_residuals_diagnostics(residuals, preds, date, 'TFT')


