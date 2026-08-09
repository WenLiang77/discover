import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def covid_forecast(train_values, horizon, **kwargs):
    train_values = np.array(train_values)
    N, T = train_values.shape
    forecasted_values = np.zeros((horizon, N))
    for region in range(N):
        try:
            model = SARIMAX(train_values[region], order=(5, 1, 0), seasonal_order=(1, 1, 1, 7))
            results = model.fit(disp=False)
            forecast = results.get_forecast(steps=horizon).predicted_mean
            forecasted_values[:, region] = np.maximum(forecast, 0)
        except Exception as e:
            forecasted_values[:, region] = 0
    return forecasted_values