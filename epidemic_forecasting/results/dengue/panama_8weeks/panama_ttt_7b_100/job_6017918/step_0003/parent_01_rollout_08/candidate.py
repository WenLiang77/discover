import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

def dengue_forecast(train_values, horizon, **kwargs):
    num_regions = train_values.shape[1]
    forecasted_values = np.zeros((horizon, num_regions))
    for region in range(num_regions):
        try:
            model = SARIMAX(train_values[:, region], order=(2, 1, 0), seasonal_order=(1, 1, 1, 52))
            model_fit = model.fit(disp=False)
            forecast = model_fit.forecast(steps=horizon)
            forecasted_values[:, region] = np.maximum(forecast, 0)
        except Exception as _:
            try:
                model = ExponentialSmoothing(train_values[:, region], trend=None, seasonal='additive', seasonal_periods=52)
                model_fit = model.fit(disp=False)
                forecast = model_fit.forecast(steps=horizon)
                forecasted_values[:, region] = np.maximum(forecast, 0)
            except Exception as _:
                try:
                    model = ARIMA(train_values[:, region], order=(2, 1, 0))
                    model_fit = model.fit(disp=False)
                    forecast = model_fit.forecast(steps=horizon)
                    forecasted_values[:, region] = np.maximum(forecast, 0)
                except Exception as _:
                    rolling_mean = np.convolve(train_values[:, region], np.ones(horizon) / horizon, mode='valid')
                    padded_rolling_mean = np.pad(rolling_mean, (0, horizon - len(rolling_mean)), mode='constant', constant_values=np.nan)
                    forecasted_values[:, region] = np.maximum(padded_rolling_mean, 0)
    return forecasted_values