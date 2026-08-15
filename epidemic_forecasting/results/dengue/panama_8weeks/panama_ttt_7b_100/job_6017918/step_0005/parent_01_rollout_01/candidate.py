import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error, symmetric_mean_absolute_percentage_error

def dengue_forecast(train_values, horizon, **kwargs):
    forecast = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            train_values[:, i] = np.where(train_values[:, i] <= 0, np.nan, train_values[:, i])
            _, p_value = acorr_ljungbox(train_values[:, i], lags=[1], boxpierce=True)
            if p_value < 0.05:
                train_values[:, i] = np.diff(train_values[:, i], n=1)
            model = auto_arima(train_values[:, i], start_p=1, start_q=1, max_p=5, max_q=5, m=4, stepwise=True, suppress_warnings=True)
            results = model.fit(train_values[:, i])
            forecast[:, i] = results.forecast(steps=horizon, alpha=0.05)[0]
            forecast[:, i] = np.maximum(forecast[:, i], 0)
        except Exception as e:
            try:
                scaler = StandardScaler()
                X_train = np.arange(len(train_values)).reshape(-1, 1)
                y_train = train_values[:, i].reshape(-1, 1)
                scaler.fit(X_train)
                X_train_scaled = scaler.transform(X_train)
                y_train_scaled = scaler.transform(y_train)
                lin_reg = LinearRegression()
                lin_reg.fit(X_train_scaled, y_train_scaled)
                X_forecast = np.arange(len(train_values), len(train_values) + horizon).reshape(-1, 1)
                X_forecast_scaled = scaler.transform(X_forecast)
                forecast_scaled = lin_reg.predict(X_forecast_scaled)
                forecast[:, i] = scaler.inverse_transform(forecast_scaled)
                forecast[:, i] = np.maximum(forecast[:, i], 0)
            except Exception as e:
                regional_means = np.nanmean(train_values[:, train_values[:, i] > 0], axis=1)
                forecast[:, i] = np.interp(range(horizon), [0, len(regional_means) - 1], regional_means)
    return forecast