import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def covid_forecast(train_values, horizon, **kwargs):
    forecasts = []
    for i in range(train_values.shape[1]):
        region_series = train_values[:, i]
        try:
            model = SARIMAX(region_series, order=(1, 1, 1), seasonal_order=(0, 1, 0, 7))
            fitted_model = model.fit(disp=False)
        except Exception as e:
            alpha = 0.1
            forecast = [np.mean(region_series)] * (horizon + len(region_series))
            for t in range(len(region_series), len(region_series) + horizon):
                forecast[t] = alpha * region_series[-1] + (1 - alpha) * forecast[t - 1]
            forecasts.append(np.clip(forecast[len(region_series):], 0, None))
            continue
        forecast = fitted_model.forecast(steps=horizon)
        forecasts.append(np.clip(forecast, 0, None))
    return np.array(forecasts).T