import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def dengue_forecast(train_values, horizon, **kwargs):
    """
    Forecast dengue incidence using SARIMAX models.
    
    Parameters:
    - train_values: np.array of shape (T, N), training data with T time steps and N regions.
    - horizon: int, number of future time steps to predict.
    
    Returns:
    - np.array of shape (horizon, N) containing predicted dengue incidences.
    """
    n_regions = train_values.shape[1]
    forecast_values = np.zeros((horizon, n_regions))
    for region in range(n_regions):
        try:
            model = SARIMAX(train_values[:, region], order=(1, 1, 0), seasonal_order=(1, 1, 0, 52))
            results = model.fit(disp=False)
            forecast = results.forecast(steps=horizon)
            forecast_values[:len(forecast), region] = np.maximum(0, forecast)
        except Exception as e:
            avg_value = np.mean(train_values[:, region])
            forecast_values[:, region] = np.full(horizon, max(0, avg_value))
    return forecast_values