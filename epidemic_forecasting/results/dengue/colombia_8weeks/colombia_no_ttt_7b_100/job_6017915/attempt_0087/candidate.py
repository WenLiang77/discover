import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def dengue_forecast(train_values, horizon, random_state=None):
    train_values = np.array(train_values)
    scaler = StandardScaler()
    scaler.fit(train_values)
    scaled_train_values = scaler.transform(train_values)
    num_regions = scaled_train_values.shape[1]
    forecast_shape = (horizon, num_regions)
    forecasts = np.zeros(forecast_shape)
    for region in range(num_regions):
        model = LinearRegression(random_state=random_state)
        try:
            model.fit(np.arange(len(scaled_train_values)).reshape(-1, 1), scaled_train_values[:, region])
            future_time_points = np.arange(len(scaled_train_values), len(scaled_train_values) + horizon).reshape(-1, 1)
            future_scaled_values = model.predict(future_time_points)
            future_values = scaler.inverse_transform(future_scaled_values.reshape(1, -1))
            future_values = np.maximum(future_values, 0)
            forecasts[:, region] = future_values.flatten()
        except Exception as e:
            forecasts[:, region] = 0
    return forecasts