import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def covid_forecast(train_values, horizon, **kwargs):
    n_regions = train_values.shape[1]
    forecast_results = []
    for region in range(n_regions):
        try:
            model = SARIMAX(train_values[:, region], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            fitted_model = model.fit(disp=False)
        except Exception as e:
            mean_value = np.mean(train_values[:, region])
            forecast = np.full(horizon, mean_value)
            forecast_results.append(forecast)
            continue
        forecast = fitted_model.get_forecast(steps=horizon).predicted_mean
        forecast = np.clip(forecast, a_min=0, a_max=None)
        forecast_results.append(forecast)
    return np.array(forecast_results).T