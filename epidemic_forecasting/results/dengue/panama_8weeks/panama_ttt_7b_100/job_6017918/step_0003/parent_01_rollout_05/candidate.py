import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def dengue_forecast(train_values, horizon, **kwargs):
    """
    Forecast Dengue incidence using a SARIMA model for each region.
    
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
            model = SARIMAX(train_values[:, region], order=(2, 1, 0), seasonal_order=(1, 1, 1, 52))
            model_fit = model.fit(disp=False)
            forecast = model_fit.forecast(steps=horizon)
            forecasted_values[:, region] = np.maximum(forecast, 0)
        except Exception as e:
            rolling_mean = np.convolve(train_values[:, region], np.ones(horizon) / horizon, mode='same')
            forecasted_values[:, region] = rolling_mean[:horizon]
    return forecasted_values