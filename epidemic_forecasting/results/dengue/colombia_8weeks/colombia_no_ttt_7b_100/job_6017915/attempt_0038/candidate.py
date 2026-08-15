import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def dengue_forecast(train_values, horizon, **kwargs):
    """
    Forecast dengue incidence for multiple regions using a SARIMA model.

    Parameters:
    - train_values: np.ndarray, shape (T, N), historical dengue case counts.
    - horizon: int, number of future time steps to predict.

    Returns:
    - np.ndarray, shape (horizon, N), predicted dengue case counts.
    """
    T, N = train_values.shape
    forecasted_values = np.zeros((horizon, N))
    for i in range(N):
        try:
            model = SARIMAX(train_values[:, i], order=(1, 1, 1), seasonal_order=(1, 1, 0, 52))
            results = model.fit(disp=False)
            forecasted_values[:, i] = results.forecast(steps=horizon)
        except Exception as e:
            print(f'Failed to fit model for region {i}: {e}')
            forecasted_values[:, i] = train_values[-1, i]
    return np.maximum(forecasted_values, 0)