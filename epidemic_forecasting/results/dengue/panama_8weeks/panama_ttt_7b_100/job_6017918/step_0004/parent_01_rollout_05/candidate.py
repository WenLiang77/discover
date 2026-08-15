import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

def preprocess_data(train_values, scale=True):
    scaler = StandardScaler()
    if scale:
        scaled_train = scaler.fit_transform(train_values)
    else:
        scaled_train = train_values
    max_lag = 4
    X = []
    y = []
    for t in range(max_lag, len(scaled_train)):
        X.append(scaled_train[t - max_lag:t])
        y.append(scaled_train[t])
    return (np.array(X), np.array(y), scaler)

def fit_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def dengue_forecast(train_values, horizon, **kwargs):
    forecast = np.zeros((horizon, train_values.shape[1]))
    scaler = None
    model = None
    for i in range(train_values.shape[1]):
        try:
            if model is None or scaler is None:
                X, y, scaler = preprocess_data(train_values[:, i:i + 1])
                model = fit_model(X, y)
            future_inputs = np.vstack([train_values[-max_lag:, i:i + 1], np.full((horizon, 1), train_values[-1, i])])
            for t in range(horizon):
                next_input = future_inputs[max_lag + t]
                next_output = model.predict(next_input.reshape(1, -1))
                future_inputs[max_lag + t] = next_output
                next_output_original = scaler.inverse_transform(next_output.reshape(-1, 1))[0, 0]
                next_output_original = max(next_output_original, 0)
                forecast[t, i] = next_output_original
        except Exception as e:
            forecast[:, i] = train_values[-1, i]
    return forecast