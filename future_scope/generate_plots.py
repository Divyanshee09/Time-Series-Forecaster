import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from forecaster import FutureScopeForecaster
import numpy as np
import os

def create_tmlr_plots():
    os.makedirs('plots', exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Benchmark Metrics Plot
    if os.path.exists('benchmarks.csv'):
        results_df = pd.read_csv('benchmarks.csv')
        plt.figure(figsize=(10, 6))
        sns.barplot(data=results_df, x='model', y='RMSE', hue='dataset')
        plt.title('RMSE Across Models and Datasets (TMLR Benchmark)', fontsize=14)
        plt.ylabel('Root Mean Squared Error')
        plt.xlabel('Model')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('plots/benchmark_metrics.png', dpi=300)
        plt.close()
        print("Generated plots/benchmark_metrics.png")

    # 2. Forecast Comparison Plot (using one dataset)
    df = pd.read_csv('data/m4_hourly.csv')
    fsf = FutureScopeForecaster(target_col='target', datetime_col='timestamp')
    fsf.ingest_data(df.tail(1000))
    fsf.preprocess()
    fsf.select_model()
    forecast = fsf.forecast(horizon=48)
    
    plt.figure(figsize=(12, 6))
    actual = fsf.data['target'].tail(100)
    plt.plot(actual.index, actual.values, label='Actual', color='black', alpha=0.6)
    plt.plot(forecast.index, forecast['mean'], label='FutureScope Forecast', color='blue', linestyle='--')
    plt.fill_between(forecast.index, forecast['mean_ci_lower'], forecast['mean_ci_upper'], color='blue', alpha=0.2)
    plt.title('FutureScope Forecast with 95% CI (M4 Hourly)', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/forecast_comparison.png', dpi=300)
    plt.close()
    print("Generated plots/forecast_comparison.png")

    # 3. Residual Diagnostics Plot
    fig = fsf.diagnostics()
    # Plotly to static image requires kaleido, which might not be in the env.
    # I'll create a matplotlib version for safety to ensure it's "TMLR-standard".
    resid = fsf.model_fit.resid
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(resid, kde=True)
    plt.title('Residual Distribution')
    plt.subplot(1, 2, 2)
    plt.plot(resid)
    plt.title('Residuals Over Time')
    plt.tight_layout()
    plt.savefig('plots/residual_diagnostics.png', dpi=300)
    plt.close()
    print("Generated plots/residual_diagnostics.png")

if __name__ == "__main__":
    create_tmlr_plots()
