import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def dengue_forecast(train_values, horizon, **kwargs):
    forecast = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            result = adfuller(train_values[:, i])
            if result[1] > 0.05:
                train_values[:, i] = np.diff(train_values[:, i])
            model = SARIMAX(train_values[:, i], order=(1, 1, 0), seasonal_order=(1, 1, 0, 4))
            results = model.fit(disp=False)
            forecast[:, i] = results.get_forecast(steps=horizon).predicted_mean
        except Exception as e:
            forecast[:, i] = np.mean(train_values[:, i]) * np.ones(horizon)
    forecast = np.maximum(forecast, 0)
    return forecast