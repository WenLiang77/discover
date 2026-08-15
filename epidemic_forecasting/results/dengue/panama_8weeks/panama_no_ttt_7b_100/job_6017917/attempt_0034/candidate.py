import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def dengue_forecast(train_values, horizon, **kwargs):
    """
    Forecasts dengue incidence based on historical data.
    
    :param train_values: A (T, N) numpy array of historical dengue case counts.
    :param horizon: An integer indicating the number of future time steps to predict.
    :return: A (horizon, N) numpy array containing the predicted dengue case counts.
    """
    num_regions = train_values.shape[1]
    predictions = np.zeros((horizon, num_regions))
    for region in range(num_regions):
        try:
            model = SARIMAX(train_values[:, region], order=(1, 1, 1), seasonal_order=(1, 1, 1, 52))
            results = model.fit(disp=False)
            forecast = results.get_forecast(steps=horizon).predicted_mean
            forecast = np.maximum(forecast, 0)
            predictions[:, region] = forecast
        except Exception as e:
            alpha = 0.1
            smoothed_values = np.convolve(train_values[:, region], [alpha] * len(train_values[:, region]), mode='valid')
            smoothed_values = np.pad(smoothed_values, (0, len(train_values[:, region]) - len(smoothed_values)), mode='constant', constant_values=(smoothed_values[-1],))
            predictions[:, region] = smoothed_values[-horizon:]
    return predictions