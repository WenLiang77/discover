import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

def covid_forecast(train_values, horizon, **kwargs):
    forecasted_values = np.zeros((horizon, train_values.shape[1]))
    for region in range(train_values.shape[1]):
        region_data = train_values[:, region]
        region_data += 1e-06
        try:
            model = SARIMAX(region_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            fitted_model = model.fit(disp=False)
            forecast = fitted_model.get_forecast(steps=horizon).predicted_mean
        except Exception as _:
            try:
                model = ExponentialSmoothing(region_data, trend='add', seasonal='mul')
                fitted_model = model.fit()
                forecast = fitted_model.forecast(steps=horizon)
            except Exception as _:
                forecast = np.full(horizon, np.mean(region_data))
        forecasted_values[:, region] = np.maximum(forecast, 0)
    return forecasted_values