import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from scipy.signal import savgol_filter

def dengue_forecast(train_values, horizon, **kwargs):
    forecast = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            diff_series = train_values[:, i].diff().dropna()
            if len(diff_series) < 2:
                window_size = 4
                smoothed_series = savgol_filter(train_values[:, i], window_length=window_size, polyorder=1)
                forecast[:, i] = np.append(np.mean(smoothed_series[:window_size]), smoothed_series[window_size:])
            else:
                model = SARIMAX(train_values[:, i], order=(1, 1, 0), seasonal_order=(1, 1, 0, 4))
                results = model.fit(disp=False)
                forecast[:, i] = results.forecast(steps=horizon)
            if np.allclose(forecast[:, i], forecast[0, i]):
                smooth_model = ExponentialSmoothing(train_values[:, i], trend='add', seasonal='add', seasonal_periods=4)
                smooth_results = smooth_model.fit(disp=False)
                forecast[:, i] = smooth_results.forecast(steps=horizon)
            forecast[:, i] = np.maximum(forecast[:, i], 0)
        except Exception as e:
            reg_model = LinearRegression()
            reg_model.fit(np.arange(len(train_values)).reshape(-1, 1), train_values[:, i])
            forecast[:, i] = reg_model.predict(np.arange(len(train_values), len(train_values) + horizon).reshape(-1, 1))
            forecast[:, i] = np.maximum(forecast[:, i], 0)
    return forecast