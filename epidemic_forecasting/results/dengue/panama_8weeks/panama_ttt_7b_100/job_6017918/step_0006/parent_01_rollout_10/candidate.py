import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

def dengue_forecast(train_values, horizon, **kwargs):
    forecast = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            if np.any(train_values[:, i] <= 0):
                train_values[:, i] = np.where(train_values[:, i] <= 0, np.nan, train_values[:, i])
            if not np.all(np.isnan(train_values[:, i])):
                model = ExponentialSmoothing(train_values[:, i], trend='add', seasonal='add', seasonal_periods=4, initialization_method='estimated')
                fit_model = model.fit(use_boxcox=True)
                forecast[:, i] = fit_model.forecast(steps=horizon)
            else:
                regional_means = np.nanmean(train_values[:, train_values[:, i] > 0], axis=1)
                forecast[:, i] = np.interp(range(horizon), [0, len(regional_means) - 1], regional_means)
        except Exception as e:
            scaler = StandardScaler()
            X_train = np.arange(len(train_values)).reshape(-1, 1)
            y_train = train_values[:, i].reshape(-1, 1)
            scaler.fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            rf_regressor = RandomForestRegressor(n_estimators=10, random_state=kwargs.get('random_state'))
            rf_regressor.fit(X_train_scaled, y_train.ravel())
            X_forecast = np.arange(len(train_values), len(train_values) + horizon).reshape(-1, 1)
            X_forecast_scaled = scaler.transform(X_forecast)
            forecast_scaled = rf_regressor.predict(X_forecast_scaled)
            forecast[:, i] = scaler.inverse_transform(forecast_scaled.reshape(-1, 1))
        forecast[:, i] = np.maximum(forecast[:, i], 0)
    return forecast