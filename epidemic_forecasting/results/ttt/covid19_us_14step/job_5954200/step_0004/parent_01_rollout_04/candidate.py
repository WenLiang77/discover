import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.exponential_smoothing.exponential_smoothing import SimpleExpSmoothing, Holt

def covid_forecast(train_values, horizon, **kwargs):
    forecasted_values = np.zeros((horizon, train_values.shape[1]))
    for region in range(train_values.shape[1]):
        region_data = train_values[:, region]
        try:
            model = SARIMAX(region_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            fitted_model = model.fit(disp=False)
            forecast = fitted_model.get_forecast(steps=horizon).predicted_mean
            forecasted_values[:, region] = np.maximum(forecast, 0)
        except Exception as _:
            pass
        if np.any(np.isnan(forecasted_values[:, region])):
            try:
                model = ARIMA(region_data, order=(1, 1, 1))
                fitted_model = model.fit(disp=False)
                forecast = fitted_model.get_forecast(steps=horizon).predicted_mean
                forecasted_values[:, region] = np.maximum(forecast, 0)
            except Exception as _:
                pass
        if np.any(np.isnan(forecasted_values[:, region])):
            try:
                model = SimpleExpSmoothing(region_data, initial_value=region_data[0])
                fitted_model = model.fit(smoothing_level=0.2, optimized=False)
                forecast = fitted_model.forecast(steps=horizon)
                forecasted_values[:, region] = np.maximum(forecast, 0)
            except Exception as _:
                pass
        if np.any(np.isnan(forecasted_values[:, region])):
            forecasted_values[np.isnan(forecasted_values[:, region]), region] = 0
    return forecasted_values