import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def covid_forecast(train_values, horizon, **kwargs):
    forecasted_values = np.zeros((horizon, train_values.shape[1]))
    for region in range(train_values.shape[1]):
        region_data = train_values[:, region]
        try:
            model = ExponentialSmoothing(region_data, trend='add', seasonal=None)
            fitted_model = model.fit()
            forecast = fitted_model.forecast(steps=horizon)
            forecasted_values[:, region] = np.maximum(forecast, 0)
        except Exception as e:
            forecasted_values[:, region] = 0
    return forecasted_values