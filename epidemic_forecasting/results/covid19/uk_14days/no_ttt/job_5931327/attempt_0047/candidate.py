import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

def covid_forecast(train_values, horizon, **kwargs):
    train_values = np.maximum(train_values, 0)
    model = MultiOutputRegressor(LinearRegression())
    model.fit(train_values[:-horizon], train_values[horizon:])
    forecast = model.predict(train_values[-horizon:])
    forecast = np.maximum(forecast, 0)
    return forecast