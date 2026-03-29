import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from sklearn.metrics import root_mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

def seasonality_check(ts, target_col, seasonal_period, threshold = 0.1):
    try:
        stl = STL(ts[target_col], seasonal=seasonal_period)
        result = stl.fit()

        seasonal_variance = result.seasonal.var()
        original_variance = ts[target_col].var()
        strength = seasonal_variance / original_variance

        print(f"🔍 STL-based Seasonal Strength: {strength:.4f}")
        return strength >= threshold

    except Exception as e:
        print(f"⚠️ Seasonality detection failed: {e}")
        return False

def arima_hyperparameter_tuning(test, train, ts, target_col, p_values, d_values, q_values): 
    progress_bar = st.progress(0)
    total_iterations = len(p_values) * len(d_values) * len(q_values)
    current_iteration = 0
    train_data = train if ts.shape[1] == 1 else train[[target_col]]
    test_data = test if ts.shape[1] == 1 else test[[target_col]]

    for p in p_values:
        for d in d_values:
            for q in q_values:
                current_iteration += 1
                progress_bar.progress(min(current_iteration / total_iterations, 1.0))
                order = (p, d, q)
                try:
                    model = SARIMAX(train_data, order=order, enforce_stationarity=False, enforce_invertibility=False)
                    model_fit = model.fit(disp=False)
                    forecast = model_fit.forecast(steps=len(test_data))
                    rmse = root_mean_squared_error(test_data, forecast)
                    if rmse < best_score:
                        best_score = rmse
                        best_model = {
                            'order': order,
                            'rmse': rmse
                        }
                    print(f"SARIMA{order} RMSE={rmse:.4f}")
                except Exception as e:
                    print(f"Failed for SARIMA{order} - {e}")
                    continue

    print(f"\n✅ Best SARIMA order: {best_model['order']}x{['seasonal_order']} with RMSE: {best_model['rmse']:.4f}")
    return best_model

def sarima_hyperparameter_tuning(ts, train, test, target_col, p_values, d_values, q_values, P_values, D_values, Q_values, S_values):
    
    progress_bar = st.progress(0)
    total_iterations = len(p_values) * len(d_values) * len(q_values) * len(P_values) * len(D_values) * len(Q_values) * len(S_values)
    current_iteration = 0
    train_data = train if ts.shape[1] == 1 else train[[target_col]]
    test_data = test if ts.shape[1] == 1 else test[[target_col]]
    best_score = float("inf"), None
    warnings.filterwarnings("ignore")

    for p in p_values:
        for d in d_values:
            for q in q_values:
                for P in P_values:
                    for D in D_values:
                        for Q in Q_values:
                            for S in S_values:
                                current_iteration += 1
                                progress_bar.progress(min(current_iteration / total_iterations, 1.0))
                                order = (p, d, q)
                                seasonal_order = (P, D, Q, S)
                                try:
                                    model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
                                    model_fit = model.fit(disp=False)
                                    forecast = model_fit.forecast(steps=len(test_data))
                                    rmse = root_mean_squared_error(test_data, forecast)
                                    if rmse < best_score:
                                        best_score = rmse
                                        best_model = {
                                            'order': order,
                                            'seasonal_order': seasonal_order,
                                            'rmse': rmse
                                        }
                                    print(f"SARIMA{order}x{seasonal_order} RMSE={rmse:.4f}")
                                except Exception as e:
                                    print(f"Failed for SARIMA{order}x{seasonal_order} - {e}")
                                    continue

    print(f"\n✅ Best SARIMA order: {best_model['order']}x{['seasonal_order']} with RMSE: {best_model['rmse']:.4f}")
    return best_model

def auto_select_arima_or_sarima(ts, train, test, target_col, arima_params, sarima_params, seasonal_period):
    has_seasonality = seasonality_check(ts, target_col, seasonal_period, threshold = 0.1)
    if has_seasonality:
        st.success("✅ Strong seasonality detected. Proceeding with SARIMA tuning.")
        best_model = sarima_hyperparameter_tuning(
            ts=ts,
            train=train,
            test=test,
            target_col=target_col,
            p_values=sarima_params['p'],
            d_values=sarima_params['d'],
            q_values=sarima_params['q'],
            P_values=sarima_params['P'],
            D_values=sarima_params['D'],
            Q_values=sarima_params['Q'],
            S_values=sarima_params['S']
        )
        return best_model
    else:
        st.warning("⚠️ Weak or no seasonality detected. Proceeding with ARIMA tuning.")
        test_size = int(len(ts) * 0.2)
        train = ts.iloc[:-test_size]
        test = ts.iloc[-test_size:]
        best_model = arima_hyperparameter_tuning(
            test=test,
            train=train,
            ts=ts,
            target_col=target_col,
            p_values=arima_params['p'],
            d_values=arima_params['d'],
            q_values=arima_params['q']
        )
        return best_model
