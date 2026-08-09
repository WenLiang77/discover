import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, symmetric_mean_absolute_percentage_error

def covid_forecast(train_values, horizon, **kwargs):
    """
    Forecast COVID-19 incidence for multiple regions using a SARIMAX model.
    
    Parameters:
    - train_values: A 2D NumPy array of shape (T, N) representing historical incidence data.
    - horizon: The number of future time steps to forecast.
    
    Returns:
    - A 2D NumPy array of shape (horizon, N) containing the forecasted incidence values.
    """
    num_regions = train_values.shape[1]
    forecast_values = np.zeros((horizon, num_regions))
    for region in range(num_regions):
        try:
            model = SARIMAX(train_values[:, region], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            fitted_model = model.fit(disp=False)
            forecast = fitted_model.get_forecast(steps=horizon)
            forecast_values[:, region] = forecast.predicted_mean
            forecast_values[:, region] = np.clip(forecast_values[:, region], a_min=0, a_max=None)
        except Exception as e:
            ma = np.convolve(train_values[:, region], np.ones(horizon) / horizon, mode='valid')
            forecast_values[:len(ma), region] = ma
    return forecast_values