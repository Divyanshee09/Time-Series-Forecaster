"""
Ultra-Fast Benchmark for Immediate Results
Uses tiny datasets (N=200) for quick validation
"""

import pandas as pd
import numpy as np
import os
import time
import sys
from future_scope_fixed import FutureScopeForecaster
from prophet import Prophet
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from scipy.stats import norm
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

def dm_test(actual, pred1, pred2):
    """Diebold-Mariano test."""
    e1, e2 = actual - pred1, actual - pred2
    d = e1**2 - e2**2
    mean_d, var_d = np.mean(d), np.var(d, ddof=1)
    if var_d == 0:
        return 1.0
    dm_stat = mean_d / np.sqrt(var_d / len(d))
    return 2 * (1 - norm.cdf(np.abs(dm_stat)))

def quick_benchmark():
    """Run ultra-fast benchmark on tiny datasets."""
    print("\n" + "="*60)
    print("QUICK BENCHMARK (N=200 samples per dataset)")
    print("="*60 + "\n")

    results = []
    os.makedirs('figures_quick', exist_ok=True)

    # 3 tiny datasets
    datasets = [
        ('Hourly', 24, lambda t: 100 + 0.1*t + 20*np.sin(2*np.pi*t/24) + np.random.normal(0, 3, len(t))),
        ('Daily', 7, lambda t: 1000 + 5*t + 100*np.sin(2*np.pi*t/7) + np.random.normal(0, 20, len(t))),
        ('Monthly', 12, lambda t: 50 + 0.5*t + 10*np.sin(2*np.pi*t/12) + np.random.normal(0, 2, len(t)))
    ]

    for ds_name, period, gen_func in datasets:
        print(f"\n[{ds_name}] ", end='', flush=True)

        # Generate tiny dataset (N=200)
        t = np.arange(200)
        y = gen_func(t)
        df = pd.DataFrame({'ds': pd.date_range('2023-01-01', periods=200, freq='H'), 'y': y})

        # Split 80/20
        train = df.iloc[:160].copy()
        test = df.iloc[160:].copy()
        actual = test['y'].values
        h = 40

        # Prophet
        print("Prophet...", end='', flush=True)
        start = time.time()
        try:
            m = Prophet(yearly_seasonality=False, weekly_seasonality=(period==7), daily_seasonality=(period==24)).fit(train, algorithm='Newton')
            fc_p = m.predict(m.make_future_dataframe(h, freq='H')).iloc[-h:]['yhat'].values
            t_p = time.time() - start
            rmse_p = root_mean_squared_error(actual, fc_p)
            print(f" ✓ {rmse_p:.2f} ({t_p:.1f}s)", flush=True)
        except Exception as e:
            print(f" ✗ {e}", flush=True)
            fc_p, rmse_p, t_p = None, np.nan, np.nan

        # FutureScope Light
        print(f"  FutureScope Light...", end='', flush=True)
        start = time.time()
        try:
            fs = FutureScopeForecaster(target_col='y', datetime_col='ds', seasonal_period=period)
            fs.ingest_data(df)
            fs.preprocess(light_mode=True)
            fs.data = fs.data.iloc[:160]
            fs.select_model(mode='simple', max_order=2)  # Ultra-fast: max_order=2
            fc_fs = fs.forecast(horizon=h, use_ensemble=False)['mean'].values
            t_fs = time.time() - start
            rmse_fs = root_mean_squared_error(actual, fc_fs)
            print(f" ✓ {rmse_fs:.2f} ({t_fs:.1f}s)", flush=True)
        except Exception as e:
            print(f" ✗ {e}", flush=True)
            fc_fs, rmse_fs, t_fs = None, np.nan, np.nan

        # Stats
        dm_p = dm_test(actual, fc_fs, fc_p) if fc_p is not None and fc_fs is not None else np.nan
        sig = "✓ Not sig. different" if dm_p > 0.05 else "✗ Significantly different"
        print(f"  DM-test: p={dm_p:.3f} {sig}", flush=True)

        results.append({
            'Dataset': ds_name,
            'N': 200,
            'Prophet_RMSE': rmse_p,
            'Prophet_Time': t_p,
            'FutureScope_RMSE': rmse_fs,
            'FutureScope_Time': t_fs,
            'DM_pval': dm_p
        })

        # Quick plot
        if fc_p is not None and fc_fs is not None:
            plt.figure(figsize=(10, 4))
            plt.plot(actual, 'k-', label='Actual', linewidth=2)
            plt.plot(fc_p, 'r--', label='Prophet', alpha=0.7)
            plt.plot(fc_fs, 'b-', label='FutureScope', alpha=0.7)
            plt.title(f'{ds_name} Forecast (N=200, h=40)')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'figures_quick/{ds_name}_forecast.png', dpi=100)
            plt.close()

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv('benchmark_quick_results.csv', index=False)

    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(df_results.to_string(index=False))
    print(f"\nResults saved to: benchmark_quick_results.csv")
    print(f"Figures saved to: figures_quick/")

    return df_results

if __name__ == "__main__":
    quick_benchmark()
