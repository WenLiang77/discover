import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

def covid_forecast(train_values, horizon, **kwargs):
    train_values = np.array(train_values)
    T, N = train_values.shape
    forecasts = np.zeros((horizon, N))
    for region in range(N):
        try:
            model = SARIMAX(train_values[:, region], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            fitted_model = model.fit(disp=False)
            forecast = fitted_model.get_forecast(steps=horizon).predicted_mean
            forecast[forecast < 0] = 0
            forecasts[:, region] = forecast
        except Exception as e:
            forecasts[:, region] = 0
            print(f'Failed to fit model for region {region}: {e}')
    return forecasts