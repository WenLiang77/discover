import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def covid_forecast(train_values, horizon, **kwargs):
    """
    Forecast COVID-19 incidence for multiple regions using a SARIMAX model.

    Parameters:
    train_values (np.ndarray): A 2D NumPy array of shape (T, N) containing the training data,
                                where T is the number of historical time steps and N is the number of regions.
    horizon (int): The number of future time steps to predict.

    Returns:
    np.ndarray: A 2D NumPy array of shape (horizon, N) containing the forecasted values.
    """
    num_regions = train_values.shape[1]
    forecasts = np.zeros((horizon, num_regions))
    for region in range(num_regions):
        try:
            model = SARIMAX(train_values[:, region], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            results = model.fit(disp=False)
            forecast = results.get_forecast(steps=horizon).predicted_mean
            forecast = np.maximum(forecast, 0)
            forecast[np.isnan(forecast)] = 0
            forecast[np.isinf(forecast)] = 0
            forecasts[:, region] = forecast
        except Exception as e:
            forecasts[:, region] = 0
    return forecasts