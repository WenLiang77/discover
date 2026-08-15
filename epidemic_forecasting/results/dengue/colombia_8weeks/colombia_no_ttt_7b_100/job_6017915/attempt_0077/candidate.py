import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def dengue_forecast(train_values, horizon, **kwargs):
    forecast_values = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        region_series = train_values[:, i]
        try:
            model = ARIMA(region_series, order=(5, 1, 0))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=horizon)
            forecast_values[:, i] = forecast
            forecast_values[:, i] = np.maximum(forecast_values[:, i], 0)
        except Exception as e:
            try:
                model = ExponentialSmoothing(region_series, seasonal_periods=52, trend='add', seasonal='mul')
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=horizon)
                forecast_values[:, i] = forecast
                forecast_values[:, i] = np.maximum(forecast_values[:, i], 0)
            except Exception as e:
                forecast_values[:, i] = np.mean(region_series)
    return forecast_values