import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, symmetric_mean_absolute_percentage_error

def preprocess_data(train_values):
    scaler = StandardScaler()
    return scaler.fit_transform(train_values)

def train_model(preprocessed_data):
    regressor = MultiOutputRegressor(BayesianRidge())
    return regressor.fit(preprocessed_data, preprocessed_data)

def covid_forecast(train_values, horizon, **kwargs):
    preprocessed_data = preprocess_data(train_values)
    model = train_model(preprocessed_data)
    forecast = model.predict(np.zeros((horizon, preprocessed_data.shape[1])))
    inverse_forecast = np.clip(np.expm1(forecast), 0, None)
    assert inverse_forecast.shape == (horizon, train_values.shape[1])
    return inverse_forecast