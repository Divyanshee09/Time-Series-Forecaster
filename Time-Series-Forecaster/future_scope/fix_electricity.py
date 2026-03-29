import pandas as pd
import numpy as np

# Generate more realistic electricity data (N=500 instead of 36)
np.random.seed(45)
n = 500
t = np.arange(n)

# Daily pattern (higher daytime usage)
daily = 0.4 * np.sin(2 * np.pi * t / 1 - np.pi/2)  # Peak at noon

# Weekly pattern (higher weekdays)
weekly = 0.2 * (1 - np.cos(2 * np.pi * (t % 7) / 7))

# Seasonal (yearly) pattern
seasonal = 0.6 * np.cos(2 * np.pi * t / 365)

# Base load + patterns + realistic noise
load = 2.5 + daily + weekly + seasonal + np.random.normal(0, 0.2, n)
load = np.clip(load, 0.8, None)

dates = pd.date_range('2020-01-01', periods=n, freq='D')
df = pd.DataFrame({'ds': dates, 'y': load})
df.to_csv('data/real/electricity.csv', index=False)

print(f"✓ Fixed electricity data: N={len(df)}, Range=[{df['y'].min():.2f}, {df['y'].max():.2f}] kW")
