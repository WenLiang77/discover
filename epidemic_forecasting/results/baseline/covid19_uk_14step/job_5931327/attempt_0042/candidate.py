import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def covid_forecast(train_values, horizon, **kwargs):
    """
    Forecast COVID-19 incidence data using a SARIMAX model.
    
    Args:
        train_values (np.ndarray): Training data of shape (T, N).
        horizon (int): Number of future time steps to forecast.
        
    Returns:
        np.ndarray: Forecasted values of shape (horizon, N).
    """
    forecasts = np.zeros((horizon, train_values.shape[1]))
    for region in range(train_values.shape[1]):
        try:
            model = SARIMAX(train_values[:, region], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            fitted_model = model.fit(disp=False)
            forecast = fitted_model.forecast(steps=horizon)
            forecasts[:, region] = forecast.clip(0)
        except Exception as e:
            warnings.warn(f'Failed to fit SARIMAX model for region {region}: {e}')
            forecasts[:, region] = np.nan
    return forecasts