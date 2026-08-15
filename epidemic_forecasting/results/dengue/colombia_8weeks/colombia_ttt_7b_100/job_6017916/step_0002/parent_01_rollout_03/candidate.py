import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

def dengue_forecast(train_values, horizon, **kwargs):
    train_values = np.array(train_values)
    forecast = np.zeros((horizon, train_values.shape[1]))
    X_train, X_val, y_train, y_val = train_test_split(np.arange(len(train_values)).reshape(-1, 1), train_values, test_size=0.2, shuffle=False)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    y_train_scaled = scaler.transform(y_train)
    model = MultiOutputRegressor(LinearRegression())
    model.fit(X_train_scaled, y_train_scaled)
    X_val_scaled = scaler.transform(X_val)
    y_val_scaled = scaler.transform(y_val)
    y_pred_scaled = model.predict(X_val_scaled)
    mse = mean_squared_error(y_val_scaled, y_pred_scaled)
    mae = mean_absolute_error(y_val_scaled, y_pred_scaled)
    rmse = np.sqrt(mse)
    mase = mae / (np.mean(np.abs(y_val - np.median(y_val))) + 1e-08)
    print(f'Validation MSE: {mse}, Validation MAE: {mae}, Validation RMSE: {rmse}, Validation MASE: {mase}')
    for t in range(horizon):
        forecast[t] = model.predict(scaler.transform([[len(train_values) + t]]))[0]
    forecast = np.maximum(forecast, 0)
    return forecast