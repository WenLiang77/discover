import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import PowerTransformer

def dengue_forecast(train_values, horizon, random_state=None):
    forecasts = np.zeros((horizon, train_values.shape[1]))
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 52)
    for i in range(train_values.shape[1]):
        pt = PowerTransformer(method='yeo-johnson')
        transformed_data = pt.fit_transform(train_values[:, [i]])
        try:
            model = SARIMAX(transformed_data.flatten(), order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
            model_fit = model.fit(disp=False)
            forecast_steps = model_fit.forecast(steps=horizon)
            forecast_steps = pt.inverse_transform(forecast_steps.reshape(-1, 1)).flatten()
            forecasts[:, i] = np.maximum(forecast_steps, 0)
        except Exception as e:
            window_size = min(10, train_values.shape[0])
            rolling_mean = np.convolve(train_values[:, i], np.ones(window_size) / window_size, mode='valid')
            forecast_steps = np.full(horizon, rolling_mean[-1]) if len(rolling_mean) > 0 else np.zeros(horizon)
            forecasts[:, i] = np.maximum(forecast_steps, 0)
    return forecasts