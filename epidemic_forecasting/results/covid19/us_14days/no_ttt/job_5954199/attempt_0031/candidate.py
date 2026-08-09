import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def covid_forecast(train_values, horizon, **kwargs):
    """
    Forecast COVID-19 incidence using Holt-Winters exponential smoothing
    
    Parameters:
        train_values (np.ndarray): Historical incidence data with shape (T, N)
        horizon (int): Number of future time steps to predict
        
    Returns:
        np.ndarray: Forecasted incidence data with shape (horizon, N)
    """
    forecasted = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            model = ExponentialSmoothing(train_values[:, i], trend='add', seasonal=None).fit()
            forecast = model.forecast(steps=horizon)
            forecasted[:, i] = forecast.clip(0)
        except Exception as e:
            moving_avg = np.convolve(train_values[:, i], np.ones(horizon), mode='valid') / horizon
            forecasted[:len(moving_avg), i] = moving_avg
            if len(moving_avg) < horizon:
                forecasted[len(moving_avg):, i] = moving_avg[-1]
    return forecasted