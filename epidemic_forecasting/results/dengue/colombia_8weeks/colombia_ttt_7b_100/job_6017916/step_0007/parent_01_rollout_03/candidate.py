import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

def dengue_forecast(train_values, horizon, **kwargs):
    num_regions = train_values.shape[1]
    forecasted_values = np.zeros((horizon, num_regions))
    scaler = StandardScaler()
    transformed_train_values = scaler.fit_transform(train_values)
    for region in range(num_regions):
        region_data = transformed_train_values[:, region]
        try:
            model = SARIMAX(region_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52)).fit(disp=False)
        except Exception as _:
            pass
        if isinstance(model, type(None)):
            try:
                model = ExponentialSmoothing(region_data, trend='add', seasonal='add', seasonal_periods=52).fit(disp=False)
            except Exception as _:
                pass
        if isinstance(model, type(None)):
            model = ExponentialSmoothing(region_data, trend='add').fit(disp=False)
        forecasted_region_data = model.forecast(steps=horizon)
        forecasted_values[:, region] = np.clip(scaler.inverse_transform(forecasted_region_data.reshape(-1, 1)), 0, None)
    return forecasted_values