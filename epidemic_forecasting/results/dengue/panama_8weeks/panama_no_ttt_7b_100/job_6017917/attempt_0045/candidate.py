import numpy as np
from statsmodels.tsa.arima.model import ARIMA

def dengue_forecast(train_values, horizon, **kwargs):
    """
    Perform multi-step forecasting on dengue incidence data using ARIMA models.

    Parameters:
    - train_values: numpy.ndarray of shape (T, N)
        Training data containing historical dengue case counts.
    - horizon: int
        Number of future time steps to predict.
    - **kwargs: Additional keyword arguments
        May include 'budget_s', 'random_state', 'frequency', and 'time_index'.

    Returns:
    - numpy.ndarray of shape (horizon, N)
        Forecasted dengue case counts.
    """
    num_regions = train_values.shape[1]
    forecasts = []
    for i in range(num_regions):
        region_data = train_values[:, i]
        try:
            model = ARIMA(region_data, order=(5, 1, 0))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=horizon)
            forecasts.append(forecast)
        except Exception as e:
            print(f'Failed to fit model for region {i}: {e}')
            forecasts.append(np.zeros(horizon))
    forecasts_array = np.array(forecasts).reshape(horizon, num_regions)
    forecasts_array = np.maximum(forecasts_array, 0)
    return forecasts_array