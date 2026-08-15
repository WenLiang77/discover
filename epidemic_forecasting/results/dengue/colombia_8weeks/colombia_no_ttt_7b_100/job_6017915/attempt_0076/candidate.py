import numpy as np
from statsmodels.tsa.arima.model import ARIMA

def dengue_forecast(train_values, horizon, **kwargs):
    """
    Forecast dengue incidence for multiple regions over a given horizon.
    
    Parameters:
    - train_values: numpy.ndarray, shape (T, N), historical dengue cases
    - horizon: int, number of future weeks to forecast
    
    Returns:
    - numpy.ndarray, shape (horizon, N), forecasted dengue cases
    """
    num_regions = train_values.shape[1]
    forecasts = np.zeros((horizon, num_regions))
    for region in range(num_regions):
        try:
            arima_model = ARIMA(train_values[:, region], order=(5, 1, 0))
            arima_model_fit = arima_model.fit()
            forecast_steps = arima_model_fit.forecast(steps=horizon)
            forecasts[:, region] = forecast_steps
        except Exception as e:
            print(f'Failed to fit ARIMA model for region {region}: {e}')
            forecasts[:, region] = 0
    forecasts = np.maximum(forecasts, 0)
    return forecasts