import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def covid_forecast(train_values, horizon, **kwargs):
    """
    A simple linear regression model to forecast COVID-19 incidence data.
    
    Args:
    train_values (numpy.ndarray): Training data of shape (T, N).
    horizon (int): Number of future time steps to predict.
    
    Returns:
    numpy.ndarray: Predicted values of shape (horizon, N).
    """
    T, N = train_values.shape
    scaler = StandardScaler()
    model = LinearRegression()
    X = np.arange(T).reshape(-1, 1)
    y = train_values
    predictions = np.zeros((horizon, N))
    for i in range(N):
        try:
            scaler.fit(X, y[:, i])
            X_scaled = scaler.transform(X)
            model.fit(X_scaled, y[:, i])
            future_X = np.arange(T, T + horizon).reshape(-1, 1)
            future_X_scaled = scaler.transform(future_X)
            future_predictions = model.predict(future_X_scaled)
            future_predictions[future_predictions < 0] = 0
            predictions[:, i] = future_predictions
        except Exception as e:
            avg_value = np.mean(y[:, i])
            predictions[:, i] = np.full(horizon, avg_value)
    return predictions