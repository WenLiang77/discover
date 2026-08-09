import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

def covid_forecast(train_values, horizon, **kwargs):
    """
    A multi-step ahead forecast for COVID-19 incidence data using a Seasonal ARIMA model.
    
    Parameters:
    train_values (np.ndarray): Training data of shape (T, N) where T is the number of time steps
                               and N is the number of regions.
    horizon (int): Number of future time steps to forecast.
    
    Returns:
    np.ndarray: Forecasted values of shape (horizon, N).
    """
    num_regions = train_values.shape[1]
    forecasts = np.zeros((horizon, num_regions))
    decomposed_data = {i: seasonal_decompose(train_values[:, i], model='additive', period=7) for i in range(num_regions)}
    for region in range(num_regions):
        try:
            model = SARIMAX(train_values[:, region], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            fitted_model = model.fit(disp=False)
            forecast = fitted_model.forecast(steps=horizon)
            forecasts[:, region] = forecast.clip(0)
        except Exception as e:
            sma = np.convolve(train_values[:, region], np.ones(horizon) / horizon, mode='valid')
            padded_sma = np.pad(sma, ((0, horizon - len(sma)), (0, 0)), 'constant', constant_values=sma[-1])
            forecasts[:, region] = padded_sma[:horizon].clip(0)
    return forecasts