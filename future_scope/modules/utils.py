import streamlit as st
import pandas as pd
from scipy.stats import skew
import warnings
warnings.filterwarnings("ignore")

def detect_skewness(df, column):
    """Detect if the data is skewed (right or left) for numeric columns only."""
    threshold = 0.5  # Common threshold for skewness
    if pd.api.types.is_numeric_dtype(df[column]):  # Ensure column is numeric
        if df[column].nunique() <= 1:  # Avoid NaN skewness calculation for constant values
            return False
        
        return abs(skew(df[column], nan_policy="omit")) > threshold  # Ignore NaNs without dropping

    return False  # Return False for non-numeric columns
                   
def detect_outliers(df, column):
    """Detect if the data has significant outliers using IQR (only for numeric columns)."""
    if pd.api.types.is_numeric_dtype(df[column]):  # Check if column is numeric
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))]
        return not outliers.empty  # Returns True if outliers exist
    return False  # Return False for non-numeric columns
                
def detect_trend(df, column):
    """Detect if the data has a trend using a simple moving average (only for numeric columns)."""
    if pd.api.types.is_numeric_dtype(df[column]):  # Ensure column is numeric
        if df[column].count() < 3:  # Count non-null values instead of dropping NaNs
            return False  # Not enough data points to detect a trend

        rolling_mean = df[column].rolling(window=3, min_periods=1).mean()  # Handles NaNs automatically
        return rolling_mean.is_monotonic_increasing or rolling_mean.is_monotonic_decreasing

    return False  # Return False for non-numeric columns
                
def detect_seasonality(df, column):
    """Detect seasonality using autocorrelation (only for numeric columns)."""
    if pd.api.types.is_numeric_dtype(df[column]):  # Check if column is numeric
        return df[column].autocorr(lag=12) > 0.5
    return False  # Return False for non-numeric columns