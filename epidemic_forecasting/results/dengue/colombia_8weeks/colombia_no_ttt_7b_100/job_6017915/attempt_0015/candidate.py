import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def dengue_forecast(train_values, horizon, **kwargs):
    num_regions, _ = train_values.shape
    forecast = np.zeros((horizon, num_regions))
    for region in range(num_regions):
        region_data = train_values[:, region]
        if len(region_data[region_data > 0]) < 10 or np.allclose(region_data, region_data[0]):
            forecast[:, region] = np.mean(region_data)
        else:
            try:
                sarima_model = SARIMAX(region_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52))
                sarima_fit = sarima_model.fit(disp=False)
                forecast[:, region] = sarima_fit.forecast(steps=horizon)
            except Exception as e:
                try:
                    exponential_smoothed = ExponentialSmoothing(region_data, seasonal_periods=52).fit()
                    forecast[:, region] = exponential_smoothed.forecast(steps=horizon)
                except Exception as e:
                    forecast[:, region] = np.mean(region_data)
    return np.maximum(forecast, 0)