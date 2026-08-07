import numpy as np
from statsmodels.tsa.arima.model import ARIMA

def covid_forecast(train_values, horizon, **kwargs):
    """
    Forecast COVID-19 incidence for multiple regions using ARIMA models.
    
    Parameters:
    - train_values: A 2D NumPy array of shape (T, N) containing historical incidence data.
    - horizon: An integer representing the number of future time steps to predict.
    - kwargs: Additional keyword arguments (not used in this function).
    
    Returns:
    - A 2D NumPy array of shape (horizon, N) containing the forecasted incidence values.
    """
    train_values = np.array(train_values)
    forecasted_values = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            model = ARIMA(train_values[:, i], order=(5, 1, 0))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=horizon)
            forecasted_values[:, i] = forecast.clip(0)
        except Exception as e:
            print(f'Failed to fit model for region {i}: {e}')
            forecasted_values[:, i] = 0
    return forecasted_values