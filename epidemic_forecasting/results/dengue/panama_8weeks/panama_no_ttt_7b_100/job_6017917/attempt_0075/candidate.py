import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler

def dengue_forecast(train_values, horizon, **kwargs):
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_values)
    forecasts = []
    for i in range(train_values.shape[1]):
        region_data = train_scaled[:, i]
        model = ExponentialSmoothing(region_data, trend='add', seasonal='add', seasonal_periods=52).fit(disp=False)
        forecast = model.forecast(steps=horizon)
        forecasts.append(scaler.inverse_transform(np.array([forecast]).T)[0])
    return np.array(forecasts).reshape(horizon, -1)