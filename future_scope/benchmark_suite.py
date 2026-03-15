import pandas as pd
import numpy as np
import os
import time
from future_scope_fixed import FutureScopeForecaster
from prophet import Prophet
from pmdarima import auto_arima
from xgboost import XGBRegressor
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# Random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def dm_test(actual, pred1, pred2, h=1, crit="MSE"):
    """Diebold-Mariano test implementation."""
    e1 = actual - pred1
    e2 = actual - pred2
    if crit == "MSE":
        d = e1**2 - e2**2
    else:
        d = np.abs(e1) - np.abs(e2)
    
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)
    if var_d == 0: return 1.0 # No difference
    dm_stat = mean_d / np.sqrt(var_d / len(d))
    p_value = 2 * (1 - norm.cdf(np.abs(dm_stat)))
    return p_value

def prepare_datasets():
    os.makedirs('data_benchmark', exist_ok=True)
    
    # 1. M4 Hourly Simulated (1000 samples)
    t = np.arange(1000)
    seasonal_24 = 10 * np.sin(2 * np.pi * t / 24)
    seasonal_168 = 5 * np.sin(2 * np.pi * t / 168)
    trend = 0.05 * t
    noise = np.random.normal(0, 1, 1000)
    m4_data = 100 + trend + seasonal_24 + seasonal_168 + noise
    pd.DataFrame({'ds': pd.date_range('2020-01-01', periods=1000, freq='H'), 'y': m4_data}).to_csv('data_benchmark/m4_hourly.csv', index=False)

    # 2. Electricity Small
    t_elec = np.arange(600)
    elec_data = 50 + 15 * np.sin(2 * np.pi * t_elec / 24) + np.random.normal(0, 2, 600)
    elec_df = pd.DataFrame({'ds': pd.date_range('2021-01-01', periods=600, freq='H'), 'y': elec_data})
    elec_df = elec_df.drop(np.random.choice(elec_df.index, 50, replace=False)).sort_index()
    elec_df.to_csv('data_benchmark/electricity_small.csv', index=False)

    # 3. Traffic Simulated
    t_traffic = np.arange(1200)
    traffic_data = 1000 + 200 * np.sin(2 * np.pi * t_traffic / 7) + np.random.normal(0, 20, 1200)
    pd.DataFrame({'ds': pd.date_range('2010-01-01', periods=1200, freq='D'), 'y': traffic_data}).to_csv('data_benchmark/traffic.csv', index=False)

    # 4. Airline Passengers (Classic)
    from statsmodels.datasets import get_rdataset
    airline = get_rdataset("AirPassengers").data
    airline = airline.rename(columns={'value': 'y'})
    airline['ds'] = pd.date_range('1949-01-01', periods=len(airline), freq='MS')
    airline = airline[['ds', 'y']]
    airline.to_csv('data_benchmark/airline_passengers.csv', index=False)

    # 5. Synthetic Irregular (Gaps + Outliers)
    t_synth = np.arange(1000)
    synth_data = 200 + 50 * np.cos(2 * np.pi * t_synth / 50) + np.random.normal(0, 10, 1000)
    # Add outliers
    synth_data[np.random.choice(1000, 20)] += 300
    synth_df = pd.DataFrame({'ds': pd.date_range('2022-01-01', periods=1000, freq='D'), 'y': synth_data})
    synth_df = synth_df.drop(np.random.choice(1000, 150, replace=False)).sort_index()
    synth_df.to_csv('data_benchmark/synthetic_irregular.csv', index=False)

def run_benchmarks():
    datasets = {
        'M4_hourly': ('data_benchmark/m4_hourly.csv', 24),
        'electricity': ('data_benchmark/electricity_small.csv', 24),
        'traffic': ('data_benchmark/traffic.csv', 7),
        'airline': ('data_benchmark/airline_passengers.csv', 12),
        'synthetic': ('data_benchmark/synthetic_irregular.csv', 50)
    }
    
    results = []
    os.makedirs('figures', exist_ok=True)
    
    for ds_name, (path, period) in datasets.items():
        print(f"Benchmarking {ds_name}...")
        df = pd.read_csv(path)
        df['ds'] = pd.to_datetime(df['ds'])
        N = len(df)
        
        # 80/20 split
        split_idx = int(N * 0.8)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        actual = test_df['y'].values
        
        # --- Baseline: Prophet ---
        start = time.time()
        m_prophet = Prophet(daily_seasonality=True if period==24 else False, 
                            weekly_seasonality=True if period==7 else False).fit(train_df)
        fut = m_prophet.make_future_dataframe(periods=len(test_df), freq=pd.infer_freq(df['ds']))
        fc_prophet = m_prophet.predict(fut).iloc[-len(test_df):]['yhat'].values
        time_prophet = time.time() - start
        rmse_prophet = root_mean_squared_error(actual, fc_prophet)
        
        # --- Baseline: AutoARIMA ---
        start = time.time()
        m_arima = auto_arima(train_df['y'], seasonal=True, m=period, error_action='ignore', trace=True)
        fc_arima = m_arima.predict(n_periods=len(test_df)).values
        time_arima = time.time() - start
        rmse_arima = root_mean_squared_error(actual, fc_arima)

        # --- Baseline: XGBoost ---
        start = time.time()
        # simplified lagged XGB
        train_y = train_df['y'].values
        train_X = np.stack([np.roll(train_y, i) for i in range(1, 13)], axis=1)[12:]
        train_target = train_y[12:]
        m_xgb = XGBRegressor().fit(train_X, train_target)
        # recursive
        curr = train_y[-12:][::-1]
        fc_xgb = []
        for _ in range(len(test_df)):
            p = m_xgb.predict(curr.reshape(1, -1))[0]
            fc_xgb.append(p)
            curr = np.roll(curr, 1)
            curr[0] = p
        time_xgb = time.time() - start
        rmse_xgb = root_mean_squared_error(actual, fc_xgb)

        # --- FutureScope: Full ---
        fs_full = FutureScopeForecaster(target_col='y', datetime_col='ds', seasonal_period=period)
        fs_full.ingest_data(df)
        fs_full.preprocess(light_mode=False)
        fs_full.data = fs_full.data.iloc[:split_idx] # split manually for benchmark
        start = time.time()
        fs_full.select_model(mode='ensemble')
        fc_fs_full = fs_full.forecast(horizon=len(test_df), use_ensemble=True)['mean'].values
        time_fs_full = time.time() - start
        rmse_fs_full = root_mean_squared_error(actual, fc_fs_full)

        # --- FutureScope: Light ---
        fs_light = FutureScopeForecaster(target_col='y', datetime_col='ds', seasonal_period=period)
        fs_light.ingest_data(df)
        fs_light.preprocess(light_mode=True)
        fs_light.data = fs_light.data.iloc[:split_idx]
        start = time.time()
        fs_light.select_model(mode='simple')
        fc_fs_light = fs_light.forecast(horizon=len(test_df), use_ensemble=False)['mean'].values
        time_fs_light = time.time() - start
        rmse_fs_light = root_mean_squared_error(actual, fc_fs_light)
        
        # --- FutureScope: Hybrid ---
        fs_hybrid = FutureScopeForecaster(target_col='y', datetime_col='ds', seasonal_period=period)
        fs_hybrid.ingest_data(df)
        fs_hybrid.preprocess()
        fs_hybrid.data = fs_hybrid.data.iloc[:split_idx]
        start = time.time()
        fs_hybrid.select_model()
        fs_hybrid.fit_hybrid_transformer(epochs=5)
        fc_fs_hybrid = fs_hybrid.forecast_hybrid(horizon=len(test_df))['mean'].values
        time_fs_hybrid = time.time() - start
        rmse_fs_hybrid = root_mean_squared_error(actual, fc_fs_hybrid)

        # DM Test vs Prophet
        dm_p = dm_test(actual, fc_fs_light, fc_prophet)

        results.append({
            'Dataset': ds_name,
            'N': N,
            'Prophet_RMSE': rmse_prophet,
            'AutoARIMA_RMSE': rmse_arima,
            'XGBoost_RMSE': rmse_xgb,
            'FutureScope_Full': rmse_fs_full,
            'FutureScope_Light': rmse_fs_light,
            'FutureScope_Hybrid': rmse_fs_hybrid,
            'DM_pval_vs_Prophet': dm_p,
            'Time_s': time_fs_full
        })
        
        # Plot Forecast for this dataset
        plt.figure(figsize=(10, 5))
        plt.plot(actual, label='Actual', color='black')
        plt.plot(fc_fs_light, label='FutureScope_Light', color='blue')
        plt.plot(fc_prophet, label='Prophet', color='red', alpha=0.5)
        plt.title(f'Forecast Comparison: {ds_name}')
        plt.legend()
        plt.savefig(f'figures/{ds_name}_forecast.png')
        plt.close()

    results_df = pd.DataFrame(results)
    results_df.to_csv('benchmarks.csv', index=False)
    
    # Global Plots
    # 1. RMSE Bar Chart
    plt.figure(figsize=(12, 6))
    results_df.plot(x='Dataset', y=['Prophet_RMSE', 'AutoARIMA_RMSE', 'FutureScope_Light'], kind='bar')
    plt.title('RMSE Comparison across Datasets')
    plt.ylabel('RMSE')
    plt.savefig('figures/rmse_comparison.png')
    plt.close()
    
    # 2. Time Comparison
    plt.figure(figsize=(10, 5))
    sns.barplot(data=results_df, x='Dataset', y='Time_s')
    plt.title('FutureScope Computation Time')
    plt.savefig('figures/scaling.png')
    plt.close()

    print(results_df.to_string())
    return results_df

if __name__ == "__main__":
    prepare_datasets()
    run_benchmarks()
