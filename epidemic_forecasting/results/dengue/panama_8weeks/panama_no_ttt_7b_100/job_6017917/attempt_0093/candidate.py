import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def dengue_forecast(train_values, horizon, **kwargs):
    """
    Forecast dengue cases using SARIMA model
    
    Parameters:
    - train_values: (T, N) numpy array of historical dengue case counts
    - horizon: int, number of future time steps to predict
    
    Returns:
    - (horizon, N) numpy array of predicted dengue case counts
    """
    T, N = train_values.shape
    forecast = np.zeros((horizon, N))
    for region in range(N):
        try:
            model = SARIMAX(train_values[:, region], order=(1, 1, 1), seasonal_order=(1, 1, 0, 52))
            results = model.fit(disp=False)
            forecast[:, region] = results.forecast(steps=horizon)
        except Exception as e:
            print(f'Failed to fit model for region {region}: {e}')
            forecast[:, region] = np.nanmean(train_values, axis=0)
    forecast = np.maximum(forecast, 0)
    return forecast