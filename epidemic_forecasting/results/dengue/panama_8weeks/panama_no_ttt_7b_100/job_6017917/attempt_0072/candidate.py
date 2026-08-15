import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error, symmetric_mean_absolute_percentage_error

def dengue_forecast(train_values, horizon, **kwargs):
    num_regions = train_values.shape[1]
    forecasted_values = np.zeros((horizon, num_regions))
    for region in range(num_regions):
        region_data = train_values[:, region]
        if np.allclose(region_data, region_data[0]):
            forecasted_values[:horizon, region] = region_data[0]
        else:
            try:
                model = ExponentialSmoothing(region_data, trend='add', seasonal='mul', seasonal_periods=52)
                fitted_model = model.fit(use_box_cox=True)
                forecast = fitted_model.forecast(steps=horizon)
                forecasted_values[:horizon, region] = forecast
            except Exception as e:
                window_size = min(10, len(region_data))
                sma = np.convolve(region_data, np.ones(window_size) / window_size, mode='valid')
                forecasted_values[:len(sma), region] = sma
                forecasted_values[len(sma):, region] = region_data[-1]
    return forecasted_values