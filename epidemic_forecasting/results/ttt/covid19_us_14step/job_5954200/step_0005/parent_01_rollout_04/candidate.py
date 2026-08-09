import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

def covid_forecast(train_values, horizon, **kwargs):
    forecasted_values = np.zeros((horizon, train_values.shape[1]))
    for region in range(train_values.shape[1]):
        region_data = train_values[:, region]
        region_data_cleaned = np.where(region_data == 0, 1e-06, region_data)
        try:
            model = SARIMAX(region_data_cleaned, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            fitted_model = model.fit(disp=False)
            forecast = fitted_model.get_forecast(steps=horizon).predicted_mean
        except Exception as _:
            try:
                model = ExponentialSmoothing(region_data_cleaned, trend='add', seasonal=None)
                fitted_model = model.fit()
                forecast = fitted_model.forecast(steps=horizon)
            except Exception as _:
                forecast = np.mean(region_data_cleaned[-10:], axis=0) * np.ones(horizon)
        forecasted_values[:, region] = np.maximum(forecast, 0)
    return forecasted_values