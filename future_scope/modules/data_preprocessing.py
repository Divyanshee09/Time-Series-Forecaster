import streamlit as st
import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
import warnings

warnings.filterwarnings("ignore")

def index_set(df, datetime_col, data_continous):
        
    ts = df.copy()
    if data_continous == "Daily":
        ts.set_index(datetime_col, inplace=True)  # Ensure it's a DatetimeIndex
        ts = ts.asfreq('D')  # Set frequency to daily
        ts = ts.interpolate(method='time')
        
    if data_continous == "Weekly":
        ts.set_index(datetime_col, inplace=True)  # Ensure it's a DatetimeIndex
        ts = ts.asfreq('W')  # Set frequency to weekly
        ts = ts.interpolate(method='time')      
        
    if data_continous == "Monthly":
        ts.set_index(datetime_col, inplace=True)  # Ensure it's a DatetimeIndex
        ts = ts.asfreq('M')  # Set frequency to monthly
        ts = ts.interpolate(method='time')

    if data_continous == "Quarterly":
        ts.set_index(datetime_col, inplace=True)  # Ensure it's a DatetimeIndex
        ts = ts.asfreq('Q')  # Set frequency to quarterly
        ts = ts.interpolate(method='time')

    if data_continous == "Half-Yearly":
        ts.set_index(datetime_col, inplace=True)  # Ensure it's a DatetimeIndex
        ts = ts.asfreq('2Q')  # Set frequency to half-yearly
        ts = ts.interpolate(method='time')
        
    if data_continous == "Yearly":
        ts.set_index(datetime_col, inplace=True)  # Ensure it's a DatetimeIndex
        ts = ts.asfreq('Y')  # Set frequency to yearly
        ts = ts.interpolate(method='time')
        
    return ts, datetime_col

def is_white_noise(ts, target_col, alpha=0.05):
    
    lags = min(20, len(ts[target_col]) - 1)
    result = acorr_ljungbox(ts[target_col], lags=lags, return_df=True)
    return (result["lb_pvalue"] > alpha).all()

def stationarity(ts, target_col, max_diff=2):
    diff_count = 0
    last_actual_value = ts[target_col].iloc[-1]  # Track the last value for later use
    first_actual_value = ts[target_col].iloc[0]  # Track the last value for later use
    
    # Now run adfuller on the cleaned data
    try:
        p_value = adfuller(ts[target_col].dropna())[1]
        
        while p_value > 0.05 and diff_count < max_diff:
            diff_count += 1
            ts[target_col] = ts[target_col].diff()
            ts = ts.dropna(subset=[target_col])
            # Remove infinite values again after differencing
            ts[target_col] = ts[target_col][~ts[target_col].isin([np.inf, -np.inf])]
            p_value = adfuller(ts[target_col])[1]
        
        if diff_count == 0:
            st.success("✅ Series is already stationary. No differencing needed.")
        elif p_value <= 0.05:
            st.success(f"✅ Series became stationary after {diff_count} differencing round(s).")
        else:
            st.error(f"⚠️ After {diff_count} differencing round(s), series is still not stationary (p-value: {p_value:.4f}).")
        
        return ts[target_col], last_actual_value, first_actual_value
    
    except Exception as e:
        print(f"Error in stationarity check: {e}")
        print("Check if your time series contains enough valid data points after cleaning.")
        return None, None