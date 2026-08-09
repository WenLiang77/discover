import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
from sklearn.exceptions import NotFittedError

def covid_forecast(train_values, horizon, **kwargs):
    """
    Forecast COVID-19 incidence for multiple regions using a hybrid approach combining Exponential Smoothing and SARIMAX.
    
    Parameters:
    train_values (np.array): A 2D NumPy array of shape (T, N) containing historical incidence data.
    horizon (int): The number of future time steps to forecast.
    
    Returns:
    np.array: A 2D NumPy array of shape (horizon, N) containing the predicted incidence values.
    """
    num_regions = train_values.shape[1]
    forecasts = np.zeros((horizon, num_regions))
    for region in range(num_regions):
        try:
            es_model = ExponentialSmoothing(train_values[:, region], trend='add', seasonal='add')
            es_fit = es_model.fit(disp=False)
            residuals = train_values[:, region] - es_fit.fittedvalues
            sarimax_model = SARIMAX(residuals, order=(1, 1, 0), seasonal_order=(1, 1, 0, 7)).fit(disp=False)
            residual_forecasts = sarimax_model.forecast(steps=horizon)
            forecasts[:, region] = es_fit.forecast(steps=horizon) + residual_forecasts
        except NotFittedError:
            es_model = ExponentialSmoothing(train_values[:, region], trend='add', seasonal='add')
            es_fit = es_model.fit(disp=False)
            forecasts[:, region] = es_fit.forecast(steps=horizon)
        except Exception as e:
            print(f'Exception occurred for region {region}: {e}')
            forecasts[:, region] = 0
    forecasts = np.maximum(forecasts, 0)
    return forecasts