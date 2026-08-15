import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def dengue_forecast(train_values, horizon, **kwargs):
    """
    Forecast dengue incidence using exponential smoothing for each region.
    
    Parameters:
    train_values (np.ndarray): Training data of shape (T, N), where T is the number of time steps and N is the number of regions.
    horizon (int): Number of future time steps to predict.
    
    Returns:
    np.ndarray: Forecasted values of shape (horizon, N).
    """
    n_regions = train_values.shape[1]
    forecasts = np.zeros((horizon, n_regions))
    for i in range(n_regions):
        try:
            model = ExponentialSmoothing(train_values[:, i], trend='add', seasonal=None)
            fit_model = model.fit()
            forecast = fit_model.forecast(steps=horizon)
            forecasts[:, i] = forecast
        except Exception as e:
            ma_forecast = np.convolve(train_values[:, i], np.ones(horizon) / horizon, mode='valid')
            ma_forecast = np.pad(ma_forecast, (0, horizon - len(ma_forecast)), 'constant', constant_values=(0, 0))
            forecasts[:, i] = ma_forecast
    return forecasts