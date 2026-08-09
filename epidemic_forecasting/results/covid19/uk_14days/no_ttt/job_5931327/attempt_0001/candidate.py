import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

def covid_forecast(train_values, horizon, **kwargs):
    scaler = StandardScaler()
    scaled_train_values = scaler.fit_transform(train_values)
    forecast_results = []
    for i in range(scaled_train_values.shape[1]):
        try:
            model = SARIMAX(scaled_train_values[:, i], order=(1, 1, 1), seasonal_order=(1, 1, 1, 24))
            fitted_model = model.fit(disp=False)
            forecast = fitted_model.get_forecast(steps=horizon)
            forecast_values = forecast.predicted_mean
            inverse_forecast = scaler.inverse_transform(forecast_values.reshape(-1, 1))[:, 0]
            inverse_forecast = np.maximum(inverse_forecast, 0)
            forecast_results.append(inverse_forecast)
        except Exception as e:
            forecast_results.append(np.zeros(horizon))
    return np.array(forecast_results).T