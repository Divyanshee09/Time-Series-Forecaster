import pandas as pd
import numpy as np
from future_scope_fixed import FutureScopeForecaster
import os

def test_mini():
    print("Starting mini benchmark...")
    df = pd.DataFrame({
        'ds': pd.date_range('2023-01-01', periods=100, freq='D'),
        'y': np.random.normal(0, 1, 100)
    })
    fs = FutureScopeForecaster(target_col='y', datetime_col='ds', seasonal_period=7)
    fs.ingest_data(df)
    fs.preprocess(light_mode=True)
    fs.select_model(mode='simple')
    print("Model selected.")
    fc = fs.forecast(horizon=7)
    print("Forecast done.")
    print(fc.head())
    print("Mini benchmark success.")

if __name__ == "__main__":
    test_mini()
