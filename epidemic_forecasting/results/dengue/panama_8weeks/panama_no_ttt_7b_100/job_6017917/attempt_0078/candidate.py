import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def dengue_forecast(train_values, horizon, **kwargs):
    """
    Forecast Dengue incidence using a SARIMA model.

    Parameters:
    - train_values: np.array of shape (T, N)
    - horizon: int, number of future time steps to predict

    Returns:
    - np.array of shape (horizon, N) containing the forecasted values
    """
    N = train_values.shape[1]
    forecast = np.zeros((horizon, N))
    for region in range(N):
        try:
            model = SARIMAX(train_values[:, region], order=(1, 1, 0), seasonal_order=(1, 1, 0, 52)).fit(disp=False)
            forecast[:, region] = model.forecast(steps=horizon)
        except Exception as e:
            forecast[:, region] = np.convolve(train_values[:, region], np.ones(4) / 4, mode='valid')[-horizon:]
            forecast[:, region] = np.pad(forecast[:, region], (0, max(0, horizon - len(forecast[:, region]))), 'constant', constant_values=np.nan)
    forecast = np.maximum(forecast, 0)
    return forecast