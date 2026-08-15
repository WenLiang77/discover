import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression

def dengue_forecast(train_values, horizon, **kwargs):
    """
    Forecast dengue incidence using Holt-Winters exponential smoothing and linear regression.
    
    :param train_values: A NumPy array of shape (T, N) containing historical dengue case counts.
    :param horizon: An integer representing the number of future time steps to predict.
    :return: A NumPy array of shape (horizon, N) containing the predicted dengue cases.
    """
    n_regions = train_values.shape[1]
    forecasts = np.zeros((horizon, n_regions))
    for region in range(n_regions):
        try:
            model = ExponentialSmoothing(train_values[:, region], trend='add', seasonal='mul', seasonal_periods=52)
            fit_model = model.fit(disp=False)
            forecast = fit_model.forecast(steps=horizon)
            forecasts[:, region] = forecast.clip(0)
        except Exception as e:
            try:
                model = LinearRegression()
                model.fit(np.arange(len(train_values)).reshape(-1, 1), train_values[:, region])
                forecast = model.predict(np.arange(len(train_values), len(train_values) + horizon).reshape(-1, 1))
                forecasts[:, region] = forecast.clip(0)
            except Exception as e:
                mean_value = np.mean(train_values[:, region])
                forecasts[:, region] = np.full(horizon, max(mean_value, 0))
    return forecasts