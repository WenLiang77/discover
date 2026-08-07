import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

def covid_forecast(train_values, horizon, **kwargs):
    scaler = MinMaxScaler()
    scaled_train_values = scaler.fit_transform(train_values)
    imputer = SimpleImputer(strategy='mean')
    imputed_scaled_train_values = imputer.fit_transform(scaled_train_values)
    forecast_values = []
    for i in range(imputed_scaled_train_values.shape[1]):
        try:
            model = ARIMA(imputed_scaled_train_values[:, i], order=(5, 1, 0))
            model_fit = model.fit(disp=False)
            forecast = model_fit.forecast(steps=horizon)
            inverse_forecast = scaler.inverse_transform(forecast.reshape(-1, 1))[:, 0]
            inverse_forecast = np.clip(inverse_forecast, 0, None)
            forecast_values.append(inverse_forecast)
        except Exception as e:
            print(f'Error fitting ARIMA model for region {i}: {e}')
            forecast_values.append(np.zeros(horizon))
    forecast_array = np.array(forecast_values).T
    return forecast_array