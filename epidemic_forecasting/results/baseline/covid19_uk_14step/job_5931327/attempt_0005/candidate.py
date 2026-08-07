import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

def covid_forecast(train_values, horizon, **kwargs):
    forecast = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            decomposition = seasonal_decompose(train_values[:, i], model='additive', period=365)
            sarimax_model = SARIMAX(decomposition.trend, order=(1, 1, 0), seasonal_order=(1, 1, 0, 365))
            sarimax_results = sarimax_model.fit(disp=False)
            forecast_steps = sarimax_results.get_forecast(steps=horizon)
            forecast[:, i] = forecast_steps.predicted_mean.clip(0)
        except Exception as e:
            window_size = min(30, train_values.shape[0])
            rolling_avg = np.convolve(train_values[:, i], np.ones(window_size) / window_size, mode='valid')
            padded_avg = np.pad(rolling_avg, (window_size // 2, window_size // 2), 'constant', constant_values=(np.nan, np.nan))
            forecast[:, i] = padded_avg[:horizon].clip(0)
    return forecast