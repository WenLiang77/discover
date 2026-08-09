import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

def covid_forecast(train_values, horizon, **kwargs):
    split_point = int(0.8 * len(train_values))
    X_train = train_values[:split_point]
    y_train = train_values[split_point:]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    y_train_scaled = scaler.transform(y_train)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train_scaled.flatten())
    last_observed = train_values[-1]
    last_observed_scaled = scaler.transform([last_observed])
    forecast_scaled = model.predict(last_observed_scaled.reshape(1, -1))
    forecast = np.zeros((horizon, train_values.shape[1]))
    for t in range(horizon):
        forecast[t] = scaler.inverse_transform(forecast_scaled).flatten()
    return forecast.clip(min=0)