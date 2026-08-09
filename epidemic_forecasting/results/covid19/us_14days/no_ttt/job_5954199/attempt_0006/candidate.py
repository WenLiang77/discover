import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, symmetric_mean_absolute_percentage_error

def decompose_and_fit(region_data, horizon):
    decomposition = seasonal_decompose(region_data, model='additive', period=7)
    model = LinearRegression()
    model.fit(range(len(decomposition.resid)), decomposition.resid)
    last_point = len(decomposition.resid) - 1
    predicted_resid = np.array([model.predict(np.array([[i + last_point + 1]]))[0] for i in range(horizon)])
    forecasted_values = decomposition.trend[-1] + decomposition.seasonal[-1] + predicted_resid
    return forecasted_values

def covid_forecast(train_values, horizon, **kwargs):
    num_regions = train_values.shape[1]
    forecasts = np.zeros((horizon, num_regions))
    for region in range(num_regions):
        region_data = train_values[:, region]
        try:
            forecast = decompose_and_fit(region_data, horizon)
            forecasts[:, region] = forecast
        except Exception as e:
            moving_avg = np.convolve(region_data, np.ones(horizon), 'valid') / horizon
            forecast = np.pad(moving_avg, (0, horizon - len(moving_avg)), mode='constant', constant_values=np.nan)
            forecast[np.isnan(forecast)] = 0
            forecasts[:, region] = forecast
    forecasts[np.isinf(forecasts)] = 0
    forecasts[forecasts < 0] = 0
    return forecasts