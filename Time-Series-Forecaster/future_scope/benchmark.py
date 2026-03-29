import pandas as pd
import numpy as np
from forecaster import FutureScopeForecaster
from pmdarima import auto_arima
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import os
import time

def create_lags(series, n_lags=12):
    df = pd.DataFrame(series)
    columns = [df.shift(i) for i in range(1, n_lags + 1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.columns = [f'lag_{i}' for i in range(1, n_lags + 1)] + ['target']
    return df.dropna()

def run_benchmark():
    datasets = {
        'wheat_small': ('data/wheat_small_irregular.csv', 'Date', 'Crop_Yield'),
        'wheat_large': ('data/wheat_large.csv', 'Date', 'Crop_Yield'),
        'electricity': ('data/electricity_seasonal.csv', 'datetime', 'value'),
        'traffic': ('data/traffic_large.csv', 'date', 'value'),
        'm4_hourly': ('data/m4_hourly.csv', 'timestamp', 'target')
    }

    results = []

    for name, (path, dt_col, target) in datasets.items():
        print(f"Benchmarking {name}...")
        df = pd.read_csv(path)
        if len(df) > 5000:
            df = df.tail(5000)
            print(f"  Subsampled to 5000 points")
        
        # 1. FutureScopeForecaster
        fsf = FutureScopeForecaster(target_col=target, datetime_col=dt_col)
        fsf.ingest_data(df)
        processed_data = fsf.preprocess()
        
        # Split (80/20)
        split_idx = int(len(processed_data) * 0.8)
        train_data = processed_data.iloc[:split_idx]
        test_data = processed_data.iloc[split_idx:]
        
        # Fit & Forecast
        start_time = time.time()
        fsf.select_model()
        fc_fsf = fsf.forecast(horizon=len(test_data))['mean']
        fsf_time = time.time() - start_time
        
        # 2. auto.arima (baseline)
        start_time = time.time()
        arima_model = auto_arima(train_data[target], seasonal=True, m=12)
        fc_arima = arima_model.predict(n_periods=len(test_data))
        arima_time = time.time() - start_time

        # 3. Prophet
        start_time = time.time()
        p_df = train_data.reset_index().rename(columns={dt_col: 'ds', target: 'y'})[['ds', 'y']]
        prophet = Prophet().fit(p_df)
        future = prophet.make_future_dataframe(periods=len(test_data), freq=fsf.freq)
        fc_prophet = prophet.predict(future).iloc[-len(test_data):]['yhat']
        prophet_time = time.time() - start_time

        # 4. XGBoost (Lagged)
        start_time = time.time()
        lagged_df = create_lags(processed_data[target])
        x_train = lagged_df.iloc[:split_idx-12].drop(columns=['target'])
        y_train = lagged_df.iloc[:split_idx-12]['target']
        xgb = XGBRegressor().fit(x_train, y_train)
        # Recursive forecast for XGB (simplified for benchmark)
        curr_lags = lagged_df.iloc[split_idx-13:split_idx-1].drop(columns=['target']).values[-1]
        fc_xgb = []
        for _ in range(len(test_data)):
            pred = xgb.predict(curr_lags.reshape(1, -1))[0]
            fc_xgb.append(pred)
            curr_lags = np.roll(curr_lags, 1)
            curr_lags[0] = pred
        xgb_time = time.time() - start_time

        # Metrics
        models = {
            'FutureScope': (fc_fsf, fsf_time),
            'AutoARIMA': (fc_arima, arima_time),
            'Prophet': (fc_prophet, prophet_time),
            'XGBoost': (fc_xgb, xgb_time)
        }

        actual = test_data[target].values
        for model_name, (preds, t) in models.items():
            results.append({
                'dataset': name,
                'model': model_name,
                'RMSE': root_mean_squared_error(actual, preds),
                'MAE': mean_absolute_error(actual, preds),
                'MAPE': mean_absolute_percentage_error(actual, preds),
                'Time': t
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv('benchmarks.csv', index=False)
    print("Benchmarks saved to benchmarks.csv")
    print(results_df.groupby('model')[['RMSE', 'MAE', 'Time']].mean())

if __name__ == "__main__":
    run_benchmark()
