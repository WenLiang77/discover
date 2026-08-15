import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def dengue_forecast(train_values, horizon, **kwargs):
    forecast = np.zeros((horizon, train_values.shape[1]))
    scaler = StandardScaler()
    imputer = SimpleImputer(strategy='mean')
    for i in range(train_values.shape[1]):
        train_imputed = imputer.fit_transform(train_values[:, i].reshape(-1, 1))
        scaled_train = scaler.fit_transform(train_imputed)
        try:
            model = SARIMAX(scaled_train, order=(1, 1, 0), seasonal_order=(1, 1, 0, 4))
            results = model.fit(disp=False)
            scaled_forecast = results.forecast(steps=horizon)
            forecast[:, i] = scaler.inverse_transform(scaled_forecast.reshape(-1, 1)).flatten()
            forecast[:, i] = np.maximum(forecast[:, i], 0)
        except Exception as e:
            forecast[:, i] = np.convolve(train_values[:, i], np.ones(horizon) / horizon, mode='valid')
            forecast = np.pad(forecast, ((0, horizon - forecast.shape[0]), (0, 0)), mode='constant', constant_values=0)[:horizon]
    return forecast