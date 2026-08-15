import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

def dengue_forecast(train_values, horizon, **kwargs):
    n_regions = train_values.shape[1]
    forecast = np.zeros((horizon, n_regions))
    for i in range(n_regions):
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(train_values[:, [i]])
        model = ARIMA(scaled_data.flatten(), order=(5, 1, 0))
        model_fit = model.fit(disp=0)
        forecast[:, i], _ = model_fit.forecast(steps=horizon)
        forecast[:, i] = scaler.inverse_transform(forecast[:, i].reshape(-1, 1)).flatten()
        forecast[:, i] = np.maximum(forecast[:, i], 0)
    return forecast