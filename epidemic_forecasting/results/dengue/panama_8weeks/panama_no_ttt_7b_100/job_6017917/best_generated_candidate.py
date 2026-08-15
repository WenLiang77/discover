import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression

def dengue_forecast(train_values, horizon, **kwargs):
    """
    Forecast future dengue incidence based on historical data.

    Parameters:
    - train_values: np.array of shape (T, N), where T is the number of historical time steps and N is the number of regions.
    - horizon: int, the number of future time steps to predict.
    - kwargs: additional keyword arguments (not used).

    Returns:
    - np.array of shape (horizon, N) containing the predicted dengue cases.
    """
    num_regions = train_values.shape[1]
    forecasts = np.zeros((horizon, num_regions))
    for region in range(num_regions):
        try:
            model = SARIMAX(train_values[:, region], order=(5, 1, 0), seasonal_order=(1, 1, 0, 7))
            result = model.fit(disp=False)
            forecast = result.forecast(steps=horizon)
            forecasts[:, region] = forecast
        except Exception as e:
            try:
                model = ARIMA(train_values[:, region], order=(5, 1, 0))
                result = model.fit(disp=False)
                forecast = result.forecast(steps=horizon)
                forecasts[:, region] = forecast
            except Exception as e:
                X = np.arange(len(train_values)).reshape(-1, 1)
                y = train_values[:, region]
                model = LinearRegression()
                model.fit(X, y)
                last_value = train_values[-1, region]
                forecast = [last_value] * horizon
                forecasts[:, region] = forecast
    forecasts = np.maximum(forecasts, 0)
    return forecasts