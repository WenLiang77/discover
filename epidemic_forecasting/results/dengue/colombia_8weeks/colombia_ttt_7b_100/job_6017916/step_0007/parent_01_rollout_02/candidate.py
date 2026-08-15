import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def dengue_forecast(train_values, horizon, **kwargs):
    num_regions = train_values.shape[1]
    forecasted_values = np.zeros((horizon, num_regions))
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    for region in range(num_regions):
        region_data = train_values[:, region]
        region_data_imputed = imputer.fit_transform(region_data.reshape(-1, 1)).flatten()
        region_data_scaled = scaler.fit_transform(region_data_imputed.reshape(-1, 1)).flatten()
        try:
            model = SARIMAX(region_data_scaled, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52)).fit(disp=False)
        except Exception as e:
            forecasted_values[:, region] = np.mean(region_data_scaled)
            continue
        forecasted_region_data_scaled = model.get_forecast(steps=horizon).predicted_mean
        forecasted_region_data = scaler.inverse_transform(forecasted_region_data_scaled.reshape(-1, 1)).flatten()
        forecasted_region_data = np.maximum(forecasted_region_data, 0)
        forecasted_values[:, region] = forecasted_region_data
    return forecasted_values