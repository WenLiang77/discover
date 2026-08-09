import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def covid_forecast(train_values, horizon, **kwargs):
    """
    A forecasting function for COVID-19 incidence data using SARIMAX model.
    
    Parameters:
    - train_values: A 2D NumPy array of shape (T, N) containing historical incidence data.
    - horizon: An integer representing the number of future time steps to predict.
    
    Returns:
    - A 2D NumPy array of shape (horizon, N) containing the forecasted incidence data.
    """
    forecasted_values = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            model = SARIMAX(train_values[:, i], order=(5, 1, 0), seasonal_order=(1, 1, 0, 7))
            fitted_model = model.fit(disp=False)
            forecast = fitted_model.forecast(steps=horizon)
            forecasted_values[:, i] = np.maximum(0, forecast)
        except Exception as e:
            forecasted_values[:, i] = 0
    return forecasted_values