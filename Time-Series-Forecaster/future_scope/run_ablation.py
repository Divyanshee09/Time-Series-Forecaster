import pandas as pd
from future_scope_fixed import FutureScopeForecaster
from sklearn.metrics import root_mean_squared_error
import numpy as np

def run_ablation():
    # Use synthetic_irregular as it benefits most from preprocessing
    df = pd.read_csv('data_benchmark/synthetic_irregular.csv')
    df['ds'] = pd.to_datetime(df['ds'])
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    actual = test_df['y'].values

    configs = [
        {'name': 'Full (All On)', 'light': False, 'skip_out': False, 'skip_trans': False},
        {'name': 'No Outliers', 'light': False, 'skip_out': True, 'skip_trans': False},
        {'name': 'No Transform', 'light': False, 'skip_out': False, 'skip_trans': True},
        {'name': 'Light Mode', 'light': True, 'skip_out': True, 'skip_trans': True},
    ]

    results = []
    for cfg in configs:
        fs = FutureScopeForecaster(target_col='y', datetime_col='ds', seasonal_period=50)
        fs.ingest_data(df)
        fs.preprocess(light_mode=cfg['light'], skip_outliers=cfg['skip_out'], skip_transform=cfg['skip_trans'])
        fs.data = fs.data.iloc[:split_idx]
        fs.select_model(mode='simple')
        preds = fs.forecast(horizon=len(test_df), use_ensemble=False)['mean'].values
        rmse = root_mean_squared_error(actual, preds)
        results.append({'Configuration': cfg['name'], 'RMSE': rmse})

    ablation_df = pd.DataFrame(results)
    ablation_df.to_csv('ablation_results.csv', index=False)
    print("Ablation results saved.")
    print(ablation_df)

if __name__ == "__main__":
    run_ablation()
