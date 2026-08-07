import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def covid_forecast(train_values, horizon, **kwargs):
    predictions = np.zeros((horizon, train_values.shape[1]))
    for region in range(train_values.shape[1]):
        try:
            model = SARIMAX(train_values[:, region], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            fitted_model = model.fit(disp=False)
            forecast = fitted_model.forecast(steps=horizon)
            predictions[:, region] = forecast.clip(0)
        except Exception as e:
            print(f'Model fitting failed for region {region}: {e}')
            predictions[:, region] = 0
    return predictions