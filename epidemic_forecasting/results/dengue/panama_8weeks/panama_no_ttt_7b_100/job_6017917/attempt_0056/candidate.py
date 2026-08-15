import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def dengue_forecast(train_values, horizon, **kwargs):
    n_regions = train_values.shape[1]
    forecast = np.zeros((horizon, n_regions))
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_values)
    for i in range(n_regions):
        region_data = train_scaled[:, i]
        try:
            model = ARIMA(region_data, order=(5, 1, 0))
            model_fit = model.fit()
            forecast[:, i] = model_fit.forecast(steps=horizon)
        except Exception as e:
            print(f'Failed to fit model for region {i}: {e}')
            forecast[:, i] = np.nan
    forecast = scaler.inverse_transform(forecast)
    forecast = np.maximum(forecast, 0)
    return forecast