import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def dengue_forecast(train_values, horizon, **kwargs):
    train_values = np.array(train_values)
    forecast = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        region_data = train_values[:, i]
        if len(region_data) >= 20:
            try:
                model = SARIMAX(region_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52))
                results = model.fit(disp=False)
                forecast[:results.nobs, i] = results.predict(start=0, end=results.nobs - 1)
                forecast[results.nobs:, i] = results.forecast(steps=horizon - results.nobs)
            except:
                pass
        if np.all(np.isnan(forecast[:, i])):
            try:
                model = ExponentialSmoothing(region_data, trend='add', seasonal='add', seasonal_periods=52)
                results = model.fit()
                forecast[:results.model.initialization_samples, i] = region_data[:results.model.initialization_samples]
                forecast[results.model.initialization_samples:, i] = results.forecast(steps=horizon - results.model.initialization_samples)
            except:
                pass
    forecast = np.maximum(forecast, 0)
    return forecast