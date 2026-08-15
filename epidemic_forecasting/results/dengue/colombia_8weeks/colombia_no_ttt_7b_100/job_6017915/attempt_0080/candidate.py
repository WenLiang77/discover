import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler

def dengue_forecast(train_values, horizon, **kwargs):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_values)
    forecasts = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        region_series = train_scaled[:, i]
        try:
            model = SARIMAX(region_series, order=(1, 1, 0), seasonal_order=(1, 1, 0, 52))
            model_fit = model.fit(disp=False)
            pred = model_fit.forecast(steps=horizon)
            pred_unscaled = scaler.inverse_transform(pred.reshape(-1, 1))[:, 0]
            pred_unscaled[pred_unscaled < 0] = 0
            pred_unscaled[np.isnan(pred_unscaled)] = 0
            forecasts[:, i] = pred_unscaled
        except Exception as e:
            ma_forecast = np.convolve(region_series, np.ones(horizon) / horizon, mode='valid')
            ma_forecast = np.pad(ma_forecast, (0, horizon - len(ma_forecast)), 'constant', constant_values=(ma_forecast[-1], 0))
            ma_forecast[ma_forecast < 0] = 0
            forecasts[:len(ma_forecast), i] = ma_forecast
    return forecasts