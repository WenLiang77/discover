import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error, symmetric_mean_absolute_percentage_error

def dengue_forecast(train_values, horizon, **kwargs):
    random_state = kwargs.get('random_state', None)
    forecast = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            model = SARIMAX(train_values[:, i], order=(2, 1, 0), seasonal_order=(2, 1, 0, 52), enforce_stationarity=False, enforce_invertibility=False, random_state=random_state).fit(disp=0)
        except Exception as e:
            continue
        pred = model.forecast(steps=horizon)
        pred[pred < 0] = 0
        forecast[:, i] = pred
    return forecast