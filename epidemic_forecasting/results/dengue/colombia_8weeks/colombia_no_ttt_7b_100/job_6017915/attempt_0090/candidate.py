import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

def dengue_forecast(train_values, horizon, **kwargs):
    n_regions, _ = train_values.shape
    forecast = np.zeros((horizon, n_regions))
    for region_idx in range(n_regions):
        region_data = train_values[:, region_idx]
        if np.std(region_data) < 1e-06:
            model = ExponentialSmoothing(region_data, trend='add', seasonal=None)
            fitted_model = model.fit()
            forecast[:, region_idx] = fitted_model.forecast(steps=horizon).clip(0)
        else:
            try:
                model = ARIMA(region_data, order=(5, 1, 0), seasonal_order=(0, 1, 0, 52))
                fitted_model = model.fit(disp=False)
                forecast[:, region_idx] = fitted_model.forecast(steps=horizon).clip(0)
            except:
                model = ExponentialSmoothing(region_data, trend='add', seasonal=None)
                fitted_model = model.fit()
                forecast[:, region_idx] = fitted_model.forecast(steps=horizon).clip(0)
    return forecast