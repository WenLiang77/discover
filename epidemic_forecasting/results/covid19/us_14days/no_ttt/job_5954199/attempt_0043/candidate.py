import numpy as np
from statsmodels.tsa.arima.model import ARIMA

def covid_forecast(train_values, horizon, **kwargs):
    """
    Multivariate COVID-19 incidence forecasting using ARIMA models.
    
    Parameters:
    - train_values: numpy.ndarray of shape (T, N), where T is the number of historical time steps and N is the number of geographical regions.
    - horizon: int, the number of future time steps to predict.
    - kwargs: additional keyword arguments, such as 'budget_s', 'random_state', 'frequency', and 'time_index'.
    
    Returns:
    - numpy.ndarray of shape (horizon, N) containing predicted future values.
    """
    train_values = np.array(train_values)
    T, N = train_values.shape
    forecasted_values = np.zeros((horizon, N))
    for i in range(N):
        try:
            model = ARIMA(train_values[:, i], order=(5, 1, 0))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=horizon)
            forecasted_values[:, i] = np.maximum(forecast, 0)
        except Exception as e:
            forecasted_values[:, i] = 0
            print(f'Failed to fit ARIMA model for region {i}: {e}')
    return forecasted_values