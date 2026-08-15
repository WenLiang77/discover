import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def dengue_forecast(train_values, horizon, **kwargs):
    train_values = np.maximum(train_values, 0)
    N = train_values.shape[1]
    forecasted_values = np.zeros((horizon, N))
    for region in range(N):
        try:
            model = ExponentialSmoothing(train_values[:, region], trend='add', seasonal=None, initialization_method='estimated')
            fit_model = model.fit(disp=False)
            forecasted_values[:, region] = fit_model.forecast(steps=horizon)
        except Exception as e:
            window_size = min(10, len(train_values))
            rolling_mean = np.convolve(train_values[:, region], np.ones(window_size) / window_size, mode='valid')
            padded_rolling_mean = np.pad(rolling_mean, (window_size - 1, 0), mode='edge')
            forecasted_values[:, region] = padded_rolling_mean[:horizon]
    forecasted_values = np.maximum(forecasted_values, 0)
    return forecasted_values