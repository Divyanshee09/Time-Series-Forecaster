import pandas as pd
import numpy as np
import os

def generate_datasets():
    os.makedirs('data', exist_ok=True)
    np.random.seed(42)

    # 1. Wheat Large (from existing dataset)
    if os.path.exists('crop_yield_dataset.csv'):
        df = pd.read_csv('crop_yield_dataset.csv')
        wheat_df = df[df['Crop_Type'] == 'Wheat'].copy()
        wheat_df['Date'] = pd.to_datetime(wheat_df['Date'])
        wheat_df = wheat_df.groupby('Date')['Crop_Yield'].mean().reset_index()
        wheat_df.to_csv('data/wheat_large.csv', index=False)
        print("Generated data/wheat_large.csv")

        # 2. Wheat Small Irregular
        wheat_small = wheat_df.sample(frac=0.3).sort_values('Date')
        wheat_small.to_csv('data/wheat_small_irregular.csv', index=False)
        print("Generated data/wheat_small_irregular.csv")

    # 3. Simulated Seasonal (Electricity-like)
    dates = pd.date_range(start='2020-01-01', periods=2000, freq='H')
    seasonal_data = 100 + 20 * np.sin(2 * np.pi * dates.hour / 24) + \
                      10 * np.sin(2 * np.pi * dates.dayofweek / 7) + \
                      np.random.normal(0, 5, len(dates))
    pd.DataFrame({'datetime': dates, 'value': seasonal_data}).to_csv('data/electricity_seasonal.csv', index=False)
    print("Generated data/electricity_seasonal.csv")

    # 4. Simulated Large (25k points)
    dates_large = pd.date_range(start='2010-01-01', periods=25000, freq='D')
    large_data = np.cumsum(np.random.normal(0.1, 1, len(dates_large))) + 500
    pd.DataFrame({'date': dates_large, 'value': large_data}).to_csv('data/traffic_large.csv', index=False)
    print("Generated data/traffic_large.csv")

    # 5. M4 Hourly Simulated
    dates_m4 = pd.date_range(start='2021-01-01', periods=1000, freq='H')
    m4_data = 50 + 0.5 * np.arange(1000) + 10 * np.sin(2 * np.pi * np.arange(1000) / 24) + np.random.normal(0, 2, 1000)
    pd.DataFrame({'timestamp': dates_m4, 'target': m4_data}).to_csv('data/m4_hourly.csv', index=False)
    print("Generated data/m4_hourly.csv")

if __name__ == "__main__":
    generate_datasets()
