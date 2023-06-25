import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from statsmodels.graphics.tsaplots import plot_acf
from scipy import stats
from statsmodels.graphics.gofplots import qqplot
from tqdm import tqdm_notebook
from statsmodels.tsa.statespace.sarimax import SARIMAX

def load_data(path = 'Ana_uloha_Data.csv', predict_start_date = "2017-01-01"):
    """
    Load data and split them into train and test data

    Args:
        path (str, optional): path to data
        predict_start_date (str, optional): date from which we want to predict YYYY-MM-DD
    
    Returns:
        pd.DataFrame: all data with time features
        pd.DataFrame: train data with time features
        pd.DataFrame: test data with time features

    """

    # Load data
    data = pd.read_csv(path, sep=',')
    data['Volume'] = data['Volume'].astype(int)
    data['log_volume'] = np.log(data['Volume'])

    # Add more features (seasonality, etc.)
    data['Date'] = pd.to_datetime(data['Date'])
    data['time'] = data.index
    data['time2'] = data['time']**2
    data['time3'] = data['time']**3
    data['time4'] = data['time']**4
    data['time5'] = data['time']**5
    data['log_time'] = data['time'].apply(lambda x: np.log(x) if x > 0 else 0)
    data['prev1_volume'] = data['Volume'].shift(1)
    data['prev2_volume'] = data['Volume'].shift(2)
    data['prev3_volume'] = data['Volume'].shift(3)
    data['prev4_volume'] = data['Volume'].shift(4)
    data['prev5_volume'] = data['Volume'].shift(5)
    data['Quarter'] = data['Date'].dt.quarter.astype('category')
    data['Month'] = data['Date'].dt.month.astype('category')
    data['Week'] = data['Date'].dt.isocalendar().week
    data['Day'] = data['Date'].dt.day.astype('category')
    data['DayOfWeek'] = data['Date'].dt.dayofweek.astype('category')

    # Split data into train and test
    train_data = data[data['Date'] < predict_start_date].copy()
    test_data = data[data['Date'] >= predict_start_date].copy()

    return data, train_data, test_data

def plot_preds_actual(date, preds, actual, title, save_name, stdev_train):
    """
    Plot predictions vs actual data with confidence interval for predictions

    Args:
        date (pd.Series): pandas series of dates
        preds (pd.Series): pandas series  of predictions
        actual (pd.Series): pandas series  of actual values
        title (str): title of plot
        save_name (str): name of file to save plot
        stdev_train (float): standard deviation of the first train data window

    """
    stdev_conf = np.multiply(1.96, stdev_train)

    conf_int = (preds - stdev_conf, preds + stdev_conf)

    plt.plot(date, preds, label='Predictions', color='red', linestyle='--', linewidth=0.5)
    plt.plot(date, actual, label='Actual', color = 'black', linewidth=0.5)
    plt.fill_between(date, conf_int[0], conf_int[1], color='blue', alpha=0.1, label = 'confidence interval')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.title(title)
    plt.savefig(save_name+'.png')


def calculate_metrics(preds, actual):
    """"
    Calculate SSE and R2 metrics
    
    Args:
        preds (pd.Series): pandas series of predictions
        actual (pd.Series): pandas series of actual values

    Returns:
        float: SSE
        float: R2
        pd.Series: residuals
    """

    residuals = preds - actual
    sse = np.sum((residuals)**2)
    r2 = r2_score(actual, preds)
    
    return sse, r2, residuals

def plot_residuals_diagnostics(residuals, preds, date, model_name):
    """
    Plot residuals diagnostics (residuals vs date, autocorrelation, residuals vs fitted values, qqplot)
    and save

    Args:
        residuals (pd.Series): residuals
        preds (pd.Series): predictions
        date (pd.Series): date
        model_name (str): name of the model

    """

    fig, ((ax1, ay1), (ax2, ay2)) = plt.subplots(2, 2, figsize=(16,16))

    ax1.scatter(date, residuals, label = 'Residuals vs Date')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Volume')
    ax1.set_title('Residuals')

    plot_acf(residuals, ax=ay1)
    ay1.set_title('Autocorrelation')
    ay1.set_xlabel('Lag')

    # plot residuals vs fitted values
    ax2.scatter(preds, residuals)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_title('Residuals vs fitted values')
    ax2.set_xlabel('Fitted values')
    ax2.set_ylabel('Residuals')

    # plot qqplot of residuals
    qqplot(residuals, stats.t, fit=True, line="45", ax=ay2)
    ay2.set_title('QQplot')


    # set title for the whole figure as Holt-Winters and remove padding between suptitle and figures
    fig.suptitle(f'Residuals diagnostics - {model_name}', fontsize=16)
    fig.tight_layout(pad=2)

    plt.savefig(f'residuals_diagnostics_{model_name}.png')


def fit_sarima(parameters_list, d, D, s, endog):
    """
    Fit SARIMA models for all paramters in parameters_list, and return results sorted by AIC 
    
    Args:
        parameters_list - list with (p, q, P, Q) tuples
        d - integration order in ARIMA model
        D - seasonal integration order
        s - length of season
        endog - the observed time series process
    
    Returns:
        results_df - dataframe with columns (p, q, P, Q) and corresponding AIC values
    """
    
    results = []
    
    for param in tqdm_notebook(parameters_list):
        try: 
            model = SARIMAX(endog, order=(param[0], d, param[1]), seasonal_order=(param[2], D, param[3], s)).fit(disp=-1, n_jobs=-1)
        except:
            continue
            
        aic = model.aic
        results.append([param, aic])
        
    result_df = pd.DataFrame(results)
    result_df.columns = ['(p,q)x(P,Q)', 'AIC']
    #Sort in ascending order
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    
    return result_df