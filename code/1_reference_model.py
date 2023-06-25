import matplotlib.pyplot as plt
import pandas as pd
import _functions as qtf
import numpy as np

plt.rcParams['figure.figsize'] = [10, 8]

#  Load data
data, train_data, test_data = qtf.load_data()

# Prepare reference model
# Reference model is a model that predicts the previous value for the next value
reference_model = data.copy()
reference_model['preds'] = reference_model['Volume'].shift(1)
reference_model_test = reference_model.iloc[test_data.index, :].copy()
reference_model_test['preds'] = reference_model_test['preds'].astype(int)

date = reference_model_test['Date'].copy()
preds = reference_model_test['preds'].copy()
actual = reference_model_test['Volume'].copy()

# Plot predictions vs actual data
qtf.plot_preds_actual(date = date, 
                      preds = preds, 
                      actual = actual, 
                      title = 'Reference model vs test data', 
                      save_name = 'reference_model'
                      )

# Plot residuals
qtf.plot_preds_actual(date = date, 
                      preds = preds, 
                      actual = actual, 
                      title = 'Residuals of reference model vs test data', 
                      save_name = 'reference_model_residuals',
                      plot_residuals=True
                      )


# Calculate R2, SSE, residuals
sse, r2, residuals = qtf.calculate_metrics(preds = preds, actual = actual)

residuals.describe()