import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

def decompose_and_fit(data, order, seasonal_order):
    decomposition = seasonal_decompose(data, model='additive')
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    try:
        model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
        fitted_model = model.fit(disp=False)
        return fitted_model
    except Exception as e:
        print(f'Failed to fit SARIMAX model: {e}')
        return None

def exponential_smoothing_fallback(data, trend='add'):
    try:
        model = ExponentialSmoothing(data, trend=trend, seasonal=None)
        fitted_model = model.fit()
        return fitted_model
    except Exception as e:
        print(f'Failed to fit Exponential Smoothing model: {e}')
        return None

def covid_forecast(train_values, horizon, **kwargs):
    forecasted_values = np.zeros((horizon, train_values.shape[1]))
    for region in range(train_values.shape[1]):
        region_data = train_values[:, region]
        decomposition = seasonal_decompose(region_data, model='additive')
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid
        sarimax_model = decompose_and_fit(region_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
        if sarimax_model:
            forecast = sarimax_model.get_forecast(steps=horizon).predicted_mean
        else:
            exp_smooth_model = exponential_smoothing_fallback(region_data, trend='add')
            if exp_smooth_model:
                forecast = exp_smooth_model.forecast(steps=horizon)
            else:
                forecast = np.full(horizon, np.nan)
        forecasted_values[:, region] = np.maximum(forecast, 0)
    return forecasted_values