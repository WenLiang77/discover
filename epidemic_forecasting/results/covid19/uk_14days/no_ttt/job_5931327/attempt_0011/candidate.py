import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

def covid_forecast(train_values, horizon, **kwargs):
    train_values = np.array(train_values)
    scaler = StandardScaler()
    scaler.fit(train_values)
    scaled_train_values = scaler.transform(train_values)
    forecasted_values = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            model = SARIMAX(scaled_train_values[:, i], order=(5, 1, 0), seasonal_order=(1, 1, 0, 7))
            results = model.fit(disp=False)
            forecast = results.get_forecast(steps=horizon)
            forecasted_mean = forecast.predicted_mean
            inverse_transformed_forecast = scaler.inverse_transform(forecasted_mean.reshape(-1, 1))[:, 0]
            forecasted_values[:, i] = inverse_transformed_forecast.clip(0)
        except Exception as e:
            forecasted_values[:, i] = 0
    return forecasted_values