import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def covid_forecast(train_values, horizon, **kwargs):
    n_regions = train_values.shape[1]
    forecasted_values = np.zeros((horizon, n_regions))
    for region in range(n_regions):
        region_data = train_values[:, region]
        try:
            model = SARIMAX(region_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            fitted_model = model.fit(disp=False)
            forecast_result = fitted_model.get_forecast(steps=horizon)
            forecasted_values[:, region] = forecast_result.predicted_mean.clip(min=0)
        except Exception as e:
            try:
                model = ExponentialSmoothing(region_data, trend='add', seasonal='mul', seasonal_periods=7)
                fitted_model = model.fit()
                forecast_result = fitted_model.forecast(horizon)
                forecasted_values[:, region] = forecast_result.clip(min=0)
            except Exception as e:
                forecasted_values[:, region] = np.mean(region_data[-10:], axis=0) * np.ones(horizon)
    return forecasted_values