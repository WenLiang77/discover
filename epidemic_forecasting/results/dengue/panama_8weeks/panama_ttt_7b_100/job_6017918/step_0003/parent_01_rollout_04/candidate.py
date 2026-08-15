import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error, symmetric_mean_absolute_percentage_error
from itertools import combinations

def check_stationarity(data):
    result = adfuller(data)
    return result[1] < 0.05

def get_seasonal_period(data):
    acf_plot = plot_acf(data, lags=50, ax=None, alpha=0.05, title=None, zero=True, vlines=True, textsize=None, label=None, marker='o', markersize=5, markerfacecolor='b', markeredgecolor='b')
    pacf_plot = plot_pacf(data, lags=50, ax=None, alpha=0.05, title=None, zero=True, vlines=True, textsize=None, label=None, marker='o', markersize=5, markerfacecolor='b', markeredgecolor='b')
    return 52

def dengue_forecast(train_values, horizon, **kwargs):
    """
    Forecast Dengue incidence using a combination of SARIMA and Exponential Smoothing models for each region.
    
    Parameters:
        train_values (np.ndarray): Historical dengue incidence data with shape (T, N).
        horizon (int): Number of future time steps to predict.
        kwargs: Additional keyword arguments (not used).
        
    Returns:
        np.ndarray: Forecasted dengue incidence data with shape (horizon, N).
    """
    num_regions = train_values.shape[1]
    forecasted_values = np.zeros((horizon, num_regions))
    for region in range(num_regions):
        try:
            data = train_values[:, region]
            if check_stationarity(data):
                seasonal_period = get_seasonal_period(data)
                model = SARIMAX(data, order=(2, 1, 0), seasonal_order=(1, 1, 1, seasonal_period))
            else:
                model = ExponentialSmoothing(data, trend='add', seasonal='mul', seasonal_periods=get_seasonal_period(data))
            model_fit = model.fit(disp=False)
            forecast = model_fit.forecast(steps=horizon)
            forecasted_values[:, region] = forecast.clip(min=0)
        except Exception as e:
            rolling_mean = np.convolve(train_values[:, region], np.ones(horizon) / horizon, mode='valid')
            padded_rolling_mean = np.pad(rolling_mean, (0, horizon - len(rolling_mean)), mode='constant', constant_values=np.nan)
            forecasted_values[:, region] = padded_rolling_mean.clip(min=0)
    return forecasted_values