import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import PowerTransformer

def covid_forecast(train_values, horizon, **kwargs):
    forecasted_values = np.zeros((horizon, train_values.shape[1]))
    for region in range(train_values.shape[1]):
        region_data = train_values[:, region]
        pt = PowerTransformer(method='yeo-johnson')
        transformed_data = pt.fit_transform(region_data.reshape(-1, 1)).flatten()
        try:
            sarimax_model = SARIMAX(transformed_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            sarimax_fitted_model = sarimax_model.fit(disp=False)
            forecast = sarimax_fitted_model.get_forecast(steps=horizon).predicted_mean
            forecasted_values[:, region] = np.maximum(pt.inverse_transform(forecast.reshape(-1, 1)), 0)
        except Exception as _:
            try:
                es_model = ExponentialSmoothing(transformed_data, trend='add', seasonal=None)
                es_fitted_model = es_model.fit()
                forecast = es_fitted_model.forecast(steps=horizon)
                forecasted_values[:, region] = np.maximum(pt.inverse_transform(forecast.reshape(-1, 1)), 0)
            except Exception as _:
                pass
    return forecasted_values