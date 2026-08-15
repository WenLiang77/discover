import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def dengue_forecast(train_values, horizon, **kwargs):
    n_regions = train_values.shape[1]
    forecast = np.zeros((horizon, n_regions))
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    for i in range(n_regions):
        ts = train_values[:, i]
        ts_imputed = imputer.fit_transform(ts.reshape(-1, 1)).flatten()
        ts_scaled = scaler.fit_transform(ts_imputed.reshape(-1, 1)).flatten()
        try:
            model = SARIMAX(ts_scaled, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52))
            model_fit = model.fit(disp=False)
            forecast[:, i] = scaler.inverse_transform(model_fit.forecast(steps=horizon).reshape(-1, 1)).flatten()
            forecast[:, i] = np.clip(forecast[:, i], 0, None)
        except Exception as e:
            forecast[:, i] = 0
    return forecast