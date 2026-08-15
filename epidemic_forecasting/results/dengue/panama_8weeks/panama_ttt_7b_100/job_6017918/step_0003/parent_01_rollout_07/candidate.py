import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def dengue_forecast(train_values, horizon, **kwargs):
    """
    Forecast Dengue incidence using multiple models including SARIMAX, ARIMA, and Exponential Smoothing.
    
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
            sarimax_model = SARIMAX(train_values[:, region], order=(2, 1, 0), seasonal_order=(1, 1, 1, 52))
            sarimax_fit = sarimax_model.fit(disp=False)
            sarimax_forecast = sarimax_fit.forecast(steps=horizon)
            arima_model = ARIMA(train_values[:, region], order=(2, 1, 0))
            arima_fit = arima_model.fit(disp=False)
            arima_forecast = arima_fit.forecast(steps=horizon)
            es_model = ExponentialSmoothing(train_values[:, region], trend='add', seasonal='multiplicative', seasonal_periods=52)
            es_fit = es_model.fit(disp=False)
            es_forecast = es_fit.forecast(steps=horizon)
            combined_forecast = (sarimax_forecast + arima_forecast + es_forecast) / 3
            forecasted_values[:, region] = np.maximum(combined_forecast, 0).astype(np.float32)
        except Exception as e:
            rolling_avg = np.convolve(train_values[:, region], np.ones(horizon) / horizon, mode='valid')
            padded_rolling_avg = np.pad(rolling_avg, (0, horizon - len(rolling_avg)), mode='constant', constant_values=np.nan)
            forecasted_values[:, region] = np.maximum(padded_rolling_avg, 0).astype(np.float32)
    return forecasted_values