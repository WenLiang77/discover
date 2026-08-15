import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy.stats import zscore

def dengue_forecast(train_values, horizon, **kwargs):
    forecast = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            diff_series = train_values[:, i].diff().dropna()
            if len(diff_series) == 0:
                continue
            models = [('SARIMAX', SARIMAX(train_values[:, i], order=(1, 1, 0), seasonal_order=(1, 1, 0, 4))), ('ARIMA', ARIMA(train_values[:, i], order=(1, 1, 0))), ('Exponential Smoothing', ExponentialSmoothing(train_values[:, i], trend='add', seasonal='mul', seasonal_periods=4))]
            results = []
            for name, model in models:
                try:
                    fit_model = model.fit(disp=False)
                    forecast_step = fit_model.forecast(steps=horizon)
                    results.append((name, fit_model, forecast_step))
                except Exception as e:
                    continue
            if not results:
                forecast[:, i] = train_values[-1, i]
                continue
            best_model, _, best_forecast = min(results, key=lambda x: x[1].aic)
            forecast[:, i] = best_forecast
        except Exception as e:
            forecast[:, i] = train_values[-1, i]
    forecast = np.maximum(forecast, 0)
    return forecast