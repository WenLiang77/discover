import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def covid_forecast(train_values, horizon, **kwargs):
    """
    Forecast COVID-19 incidence data using SARIMAX models for each region.

    Parameters:
    train_values (np.ndarray): Historical incidence data with shape (T, N).
    horizon (int): Number of future time steps to predict.
    
    Returns:
    np.ndarray: Predicted incidence data with shape (horizon, N).
    """
    N = train_values.shape[1]
    forecast_array = np.zeros((horizon, N))
    for i in range(N):
        try:
            model = SARIMAX(train_values[:, i], order=(1, 1, 0), seasonal_order=(1, 1, 0, 7))
            results = model.fit(disp=False)
            forecast = results.forecast(steps=horizon)
            forecast_array[:, i] = forecast.clip(0)
        except Exception as e:
            print(f'Failed to fit model for region {i}: {e}')
            forecast_array[:, i] = 0
    return forecast_array