import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tools.eval_measures import smape, mae, rmse, mase
from pmdarima import auto_arima

def dengue_forecast(train_values, horizon, **kwargs):
    forecasts = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            model = auto_arima(train_values[:, i], seasonal=True, m=52)
            forecast = model.predict(n_periods=horizon)
            forecast = np.maximum(forecast, 0)
            forecasts[:, i] = forecast
        except Exception as e:
            ma_value = np.mean(train_values[:, i])
            forecasts[:, i] = ma_value
    return forecasts