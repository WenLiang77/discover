import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import PowerTransformer

def covid_forecast(train_values, horizon, **kwargs):
    forecasted_values = np.zeros((horizon, train_values.shape[1]))
    for region in range(train_values.shape[1]):
        region_data = train_values[:, region]
        region_data += 1e-06
        pt = PowerTransformer(method='box-cox', standardize=False)
        region_data_transformed = pt.fit_transform(region_data.reshape(-1, 1)).flatten()
        try:
            model = SARIMAX(region_data_transformed, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            fitted_model = model.fit(disp=False)
            forecast_transformed = fitted_model.get_forecast(steps=horizon).predicted_mean
        except Exception as _:
            try:
                model = ExponentialSmoothing(region_data_transformed, trend='add', seasonal=None)
                fitted_model = model.fit()
                forecast_transformed = fitted_model.forecast(steps=horizon)
            except Exception as _:
                forecast_transformed = np.full(horizon, np.mean(region_data_transformed))
        forecast_original_scale = pt.inverse_transform(forecast_transformed.reshape(-1, 1)).flatten()
        forecasted_values[:, region] = np.maximum(forecast_original_scale, 0)
    return forecasted_values