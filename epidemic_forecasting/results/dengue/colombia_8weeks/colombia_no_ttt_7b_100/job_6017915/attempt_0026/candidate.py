import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def dengue_forecast(train_values, horizon, **kwargs):
    """
    Forecast future dengue incidence based on historical data.
    
    Parameters:
    train_values (np.array): Historical dengue incidence data of shape (T, N).
    horizon (int): Number of future time steps to predict.
    
    Returns:
    np.array: Forecasted dengue incidence of shape (horizon, N).
    """
    n_regions = train_values.shape[1]
    forecasts = np.zeros((horizon, n_regions))
    for region in range(n_regions):
        try:
            model = SARIMAX(train_values[:, region], order=(2, 1, 0), seasonal_order=(1, 1, 1, 52))
            results = model.fit(disp=False)
            forecast = results.forecast(steps=horizon)
            forecasts[:, region] = np.clip(forecast, 0, None)
        except Exception as e:
            print(f'Failed to fit SARIMA model for region {region}: {e}')
            forecasts[:, region] = np.mean(train_values[:, region]) * np.ones(horizon)
    return forecasts