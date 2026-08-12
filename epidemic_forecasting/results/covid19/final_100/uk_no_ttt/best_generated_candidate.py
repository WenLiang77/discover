import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

def preprocess_data(train_values):
    scaler = StandardScaler()
    return (scaler.fit_transform(train_values), scaler)

def fit_models(data):
    models = []
    for i in range(data.shape[1]):
        model = SARIMAX(data[:, i], order=(5, 1, 0), seasonal_order=(1, 1, 0, 7))
        model_fit = model.fit(disp=False)
        models.append(model_fit)
    return models

def covid_forecast(train_values, horizon, **kwargs):
    scaled_train, scaler = preprocess_data(train_values)
    models = fit_models(scaled_train)
    forecasted_values = np.zeros((horizon, train_values.shape[1]))
    for i, model in enumerate(models):
        forecast = model.forecast(steps=horizon)
        forecasted_values[:, i] = forecast
    forecasted_values = scaler.inverse_transform(forecasted_values)
    forecasted_values = np.maximum(0, forecasted_values)
    np.nan_to_num(forecasted_values, copy=False)
    return forecasted_values