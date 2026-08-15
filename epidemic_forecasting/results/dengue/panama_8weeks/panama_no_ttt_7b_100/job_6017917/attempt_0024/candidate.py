import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

def preprocess_data(data):
    scaler = MinMaxScaler()
    return (scaler.fit_transform(data), scaler)

def postprocess_data(forecast, scaler):
    return scaler.inverse_transform(forecast)

def fit_arima_models(train_data, order=(1, 1, 1)):
    num_regions, _ = train_data.shape
    arima_models = [ARIMA(endog=train_data[:, i], order=order).fit() for i in range(num_regions)]
    return arima_models

def fit_linear_regression(train_data):
    num_regions, _ = train_data.shape
    lr_model = LinearRegression()
    multi_lr = MultiOutputRegressor(lr_model)
    multi_lr.fit(train_data, train_data)
    return multi_lr

def dengue_forecast(train_values, horizon, **kwargs):
    scaled_train, scaler = preprocess_data(train_values)
    arima_models = fit_arima_models(scaled_train)
    if not all((model.fitted for model in arima_models)):
        print('Failed to fit ARIMA models. Using linear regression instead.')
        multi_lr = fit_linear_regression(scaled_train)
        forecast_scaled = multi_lr.predict(np.concatenate([scaled_train[-horizon:], np.zeros((horizon, scaled_train.shape[1]))]))
    else:
        forecast_scaled = np.array([model.forecast(steps=horizon) for model in arima_models])
    forecast_scaled = np.maximum(forecast_scaled, 0)
    forecast = postprocess_data(forecast_scaled, scaler)
    return forecast