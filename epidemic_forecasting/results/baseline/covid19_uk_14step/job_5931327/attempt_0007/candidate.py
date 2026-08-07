import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def covid_forecast(train_values, horizon, **kwargs):
    """
    Forecast COVID-19 incidence for multiple regions using Exponential Smoothing.
    
    Parameters:
        train_values (np.ndarray): Shape (T, N), historical incidence data for N regions over T time steps.
        horizon (int): Number of future time steps to forecast.
        
    Returns:
        np.ndarray: Shape (horizon, N), forecasts for the next horizon time steps for each region.
    """
    train_values = np.array(train_values)
    forecasted_values = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            model = ExponentialSmoothing(train_values[:, i], trend='add', seasonal=None)
            fitted_model = model.fit(disp=False)
            forecast = fitted_model.forecast(steps=horizon)
            forecasted_values[:, i] = forecast
        except Exception as e:
            print(f'Failed to fit model for region {i}: {e}')
            forecasted_values[:, i] = 0
    forecasted_values = np.maximum(forecasted_values, 0)
    return forecasted_values