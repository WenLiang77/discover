import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression

def dengue_forecast(train_values, horizon, **kwargs):
    num_regions = train_values.shape[1]
    forecasted_values = np.zeros((horizon, num_regions))
    for region in range(num_regions):
        region_data = train_values[:, region]
        if np.sum(region_data) == 0:
            forecasted_values[:, region] = 0
            continue
        try:
            model = SARIMAX(region_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52)).fit(disp=False)
            forecasted_region_data = model.get_forecast(steps=horizon).predicted_mean
        except Exception as e:
            try:
                model = ARIMA(region_data, order=(1, 1, 1)).fit(disp=False)
                forecasted_region_data = model.forecast(steps=horizon)
            except Exception as e:
                try:
                    model = LinearRegression().fit(np.arange(len(region_data)).reshape(-1, 1), region_data)
                    forecasted_region_data = model.predict(np.arange(len(region_data), len(region_data) + horizon).reshape(-1, 1))
                except Exception as e:
                    forecasted_region_data = np.mean(region_data) * np.ones(horizon)
        forecasted_region_data = np.maximum(forecasted_region_data, 0)
        forecasted_values[:, region] = forecasted_region_data
    return forecasted_values