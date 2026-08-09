import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def covid_forecast(train_values, horizon, **kwargs):
    train_values = np.array(train_values)
    N = train_values.shape[1]
    forecasts = []
    for i in range(N):
        ts = train_values[:, i]
        try:
            model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            result = model.fit(disp=False)
            forecast = result.get_forecast(steps=horizon).predicted_mean
            forecast = np.clip(forecast, a_min=0, a_max=None)
            forecasts.append(forecast)
        except Exception as e:
            print(f'Error fitting model for region {i}: {e}')
            forecasts.append(np.zeros(horizon))
    forecasts_array = np.array(forecasts)
    return forecasts_array