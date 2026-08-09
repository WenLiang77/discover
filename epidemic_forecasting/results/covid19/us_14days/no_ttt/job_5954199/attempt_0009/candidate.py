import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def covid_forecast(train_values, horizon, **kwargs):
    """
    Forecast COVID-19 incidence data using a SARIMAX model.
    
    Args:
        train_values: A 2D numpy array of shape (T, N) containing historical incidence data.
        horizon: The number of future time steps to predict.
        kwargs: Additional keyword arguments including 'budget_s', 'random_state', 'frequency', and 'time_index'.
        
    Returns:
        A 2D numpy array of shape (horizon, N) containing the forecasted incidence data.
    """
    frequency = kwargs.get('frequency', 7)
    forecasts = np.zeros((horizon, train_values.shape[1]))
    for region in range(train_values.shape[1]):
        try:
            model = SARIMAX(train_values[:, region], order=(1, 1, 1), seasonal_order=(1, 1, 1, frequency))
            fitted_model = model.fit(disp=False)
            forecast = fitted_model.forecast(steps=horizon)
            forecasts[:, region] = forecast
        except Exception as e:
            forecasts[:, region] = 0
    return forecasts.clip(min=0)