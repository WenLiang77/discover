import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

def covid_forecast(train_values, horizon, **kwargs):
    decomposed = seasonal_decompose(train_values, period=7)
    trend = decomposed.trend
    trend = np.nan_to_num(trend, nan=np.interp(np.where(np.isnan(trend))[0], np.where(~np.isnan(trend))[0], trend[~np.isnan(trend)]))
    forecast = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        alpha = 2 / (len(trend[:, i]) + 1)
        smoothed_trend = np.zeros_like(trend[:, i])
        smoothed_trend[0] = trend[0, i]
        for t in range(1, len(trend[:, i])):
            smoothed_trend[t] = alpha * trend[t, i] + (1 - alpha) * smoothed_trend[t - 1]
        extended_trend = np.concatenate([smoothed_trend, [smoothed_trend[-1]] * (horizon - len(smoothed_trend))])
        forecast[:, i] = extended_trend[:horizon]
    return forecast