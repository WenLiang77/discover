import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def covid_forecast(train_values, horizon, **kwargs):
    """
    Forecast COVID-19 incidence using a SARIMA model
    
    Parameters:
    train_values (np.ndarray): A 2D array of shape (T, N) containing historical incidence data
    horizon (int): Number of future time steps to predict
    
    Returns:
    np.ndarray: A 2D array of shape (horizon, N) containing the forecasted incidence data
    """
    num_regions = train_values.shape[1]
    forecasts = np.zeros((horizon, num_regions))
    for region_idx in range(num_regions):
        try:
            model = SARIMAX(train_values[:, region_idx], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            fitted_model = model.fit(disp=False)
            forecast = fitted_model.get_forecast(steps=horizon).predicted_mean
            forecasts[:, region_idx] = forecast.clip(0)
        except Exception as e:
            print(f'Failed to fit model for region {region_idx}: {e}')
            forecasts[:, region_idx] = np.nan
    return forecasts