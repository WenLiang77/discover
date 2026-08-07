import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def covid_forecast(train_values, horizon, **kwargs):
    train_values = np.array(train_values)
    if not isinstance(train_values, np.ndarray) or len(train_values.shape) != 2:
        raise ValueError('Input must be a 2D numpy array.')
    if horizon <= 0:
        raise ValueError('Horizon must be a positive integer.')
    decomposed = seasonal_decompose(train_values, model='additive', period=7)
    trend = decomposed.trend
    seasonal = decomposed.seasonal
    combined = trend + seasonal
    imputer = SimpleImputer(strategy='mean')
    combined_filled = imputer.fit_transform(combined)
    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined_filled)
    try:
        sarima_model = SARIMAX(combined_scaled, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)).fit(disp=False)
    except Exception as e:
        arima_model = SARIMAX(combined_scaled, order=(1, 1, 1)).fit(disp=False)
        forecast_scaled = arima_model.get_forecast(steps=horizon).forecasted_values
    else:
        forecast_scaled = sarima_model.get_forecast(steps=horizon).forecasted_values
    forecast = scaler.inverse_transform(forecast_scaled)
    forecast = np.maximum(0, forecast)
    return forecast