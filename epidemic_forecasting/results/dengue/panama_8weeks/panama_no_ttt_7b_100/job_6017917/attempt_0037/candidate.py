import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

def dengue_forecast(train_values, horizon, **kwargs):
    random_state = kwargs.get('random_state', None)
    frequency = kwargs.get('frequency', 'W')
    scaler = StandardScaler()
    imputer = KNNImputer(n_neighbors=2)
    train_scaled = scaler.fit_transform(train_values)
    train_imputed = imputer.fit_transform(train_scaled)
    forecasts = []
    for i in range(train_values.shape[1]):
        try:
            model = SARIMAX(train_imputed[:, i], order=(1, 1, 0), seasonal_order=(1, 1, 0, 52), enforce_stationarity=False, enforce_invertibility=False, random_state=random_state)
            results = model.fit(disp=False)
            forecast = results.forecast(steps=horizon)
            forecast = np.clip(scaler.inverse_transform(forecast.reshape(-1, 1)), 0, None).flatten()
        except Exception as e:
            print(f'Failed to fit model for region {i}: {e}')
            forecast = np.zeros(horizon)
        forecasts.append(forecast)
    return np.array(forecasts).T