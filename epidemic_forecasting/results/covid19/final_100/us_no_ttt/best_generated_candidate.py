import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def covid_forecast(train_values, horizon, **kwargs):
    """
    Forecast COVID-19 incidence data using a state-space model with ARIMA components.
    
    Parameters:
    - train_values: A 2D NumPy array of shape (T, N), where T is the number of historical time steps
                    and N is the number of geographical regions.
    - horizon: The number of future time steps to predict.
    
    Returns:
    - A 2D NumPy array of shape (horizon, N) containing the forecasted values.
    """
    num_regions = train_values.shape[1]
    forecasts = np.zeros((horizon, num_regions))
    for region in range(num_regions):
        try:
            model = SARIMAX(train_values[:, region], order=(1, 1, 1), seasonal_order=(0, 1, 1, 7))
            results = model.fit(disp=False)
            forecast = results.forecast(steps=horizon)
            forecasts[:, region] = forecast.clip(0)
        except Exception as e:
            print(f'Error fitting model for region {region}: {e}')
            forecasts[:, region] = 0
    return forecasts