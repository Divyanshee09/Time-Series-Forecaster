import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import plotly.graph_objects as go
from sklearn.metrics import root_mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import STL
import warnings
warnings.filterwarnings("ignore")

def arima_hyperparameter_tuning(test, train, ts, target_col, p_values, d_values, q_values): 
    
    train_data = train if ts.shape[1] == 1 else train[[target_col]]
    test_data = test if ts.shape[1] == 1 else test[[target_col]]

    results = []
    progress_bar = st.progress(0)
    total_iterations = len(p_values) * len(d_values) * len(q_values)
    current_iteration = 0
    top_n=5

    for p in p_values:
        for d in d_values:
            for q in q_values:
                current_iteration += 1
                progress_bar.progress(min(current_iteration / total_iterations, 1.0))
                order = (p, d, q)
                try:
                    model = ARIMA(train_data, 
                                    order=order, 
                                    enforce_stationarity=False, 
                                    enforce_invertibility=False)
                    model_fit = model.fit()

                    result = {
                        'order': order,
                        'model_fit': model_fit,
                    }
                    if len(train_data) <= 100:
                        result['bic'] = model_fit.bic
                        print(f"✅ ARIMA{order} | BIC: {model_fit.bic:.2f}")
                    else:
                        result['aic'] = model_fit.aic
                        print(f"✅ ARIMA{order} | AIC: {model_fit.aic:.2f}")

                    results.append(result)
                except Exception as e:
                    print(f"Failed for ARIMA{order} - {e}")
                    continue

    if not results:
        print("❌ No ARIMA models succeeded during tuning.")
        return None
    
    result_df = pd.DataFrame(results)
    
    if len(train_data) <= 100:
        top_models = result_df.nsmallest(top_n, 'bic')
    else:
        top_models = result_df.nsmallest(top_n, 'aic')
        
    best_score = float("inf")
    best_model = None
    
    for _, row in top_models.iterrows():
        try:
            order = row['order']
            model_fit = row['model_fit']
            forecast = model_fit.forecast(steps=len(test_data))
            rmse = root_mean_squared_error(test_data, forecast)
            print(f"🔍 Evaluating top model ARIMA{order} | Test RMSE: {rmse:.4f}")
            if rmse < best_score:
                best_score = rmse
                if len(train_data) <= 100:
                    best_model = {
                        'order': row['order'],
                        'bic': row['bic'],
                        'rmse': rmse,
                        'model_fit': row['model_fit']
                    }
                else:
                    best_model = {
                        'order': row['order'],
                        'aic': row['aic'],
                        'rmse': rmse,
                        'model_fit': row['model_fit']
                    }
        except Exception as e:
            print(f"❌ Test RMSE calculation failed for ARIMA{row['order']} - {e}")
            
    if best_model:
        if len(train_data) <= 100:
            print(f"\n✅ Best ARIMA order: {best_model['order']} | RMSE: {best_model['rmse']:.4f} | BIC: {best_model['bic']}")
        else:
            print(f"\n✅ Best ARIMA order: {best_model['order']} | RMSE: {best_model['rmse']:.4f} | AIC: {best_model['aic']}")   
    else:
        print("❌ ARIMA hyperparameter tuning did not return a best model.")

    return best_model

# def sarima_hyperparameter_tuning(ts, train, test, target_col, p_values, d_values, q_values, P_values, D_values, Q_values, S_values):
    train_data = train if ts.shape[1] == 1 else train[[target_col]]
    test_data = test if ts.shape[1] == 1 else test[[target_col]]

    results = []
    progress_bar = st.progress(0)
    total_iterations = len(p_values) * len(d_values) * len(q_values) * len(P_values) * len(D_values) * len(Q_values) * len(S_values)
    current_iteration = 0
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

                                    result = {
                                        'order': order,
                                        'seasonal order': seasonal_order,
                                        'model_fit': model_fit
                                    }
                                    if len(train_data) <= 100:
                                        result['bic'] = model_fit.bic
                                        print(f"✅ SARIMA{order}x{seasonal_order} | BIC: {model_fit.bic:.2f}")
                                    else:
                                        result['aic'] = model_fit.aic
                                        print(f"✅ SARIMA{order}x{seasonal_order} | AIC: {model_fit.aic:.2f}")

                                    results.append(result)
                                except Exception as e:
                                    print(f"Failed for SARIMA{order}x{seasonal_order} - {e}")
                                    continue

    result_df = pd.DataFrame(results)
    if len(train_data) <= 100:
        best_model_row = result_df.sort_values(by='bic').iloc[0]
    else:
        best_model_row = result_df.sort_values(by='aic').iloc[0]

    best_model = {
        'order': best_model_row['order'],
        'seasonal order': best_model_row['seasonal order']
    }
    st.info(f"Best hyperparameter selected for SARIMA:{best_model['order']}x{best_model['seasonal order']}")
    final_fit = best_model_row['model_fit']  # Reuse the fitted model

    # Train RMSE
    train_pred = final_fit.fittedvalues
    train_rmse = np.sqrt(root_mean_squared_error(train_data, train_pred))
    print(f"📏 Train RMSE: {train_rmse:.4f}")

    # Test RMSE
    test_rmse = None
    if test_data is not None:
        try:
            forecast = final_fit.forecast(steps=len(test_data))
            test_rmse = np.sqrt(root_mean_squared_error(test_data, forecast))
            print(f"📏 Test RMSE: {test_rmse:.4f}")
        except Exception as e:
            print(f"❌ Test RMSE Calculation Failed: {e}")

    return final_fit

def sarima_hyperparameter_tuning(ts, train, test, target_col, p_values, d_values, q_values, P_values, D_values, Q_values, S_values):
    train_data = train if ts.shape[1] == 1 else train[[target_col]]
    test_data = test if ts.shape[1] == 1 else test[[target_col]]

    results = []
    progress_bar = st.progress(0)
    total_iterations = len(p_values) * len(d_values) * len(q_values) * len(P_values) * len(D_values) * len(Q_values) * len(S_values)
    current_iteration = 0
    top_n=5
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
                                    model = SARIMAX(train_data, 
                                                    order=order, 
                                                    seasonal_order=seasonal_order, 
                                                    enforce_stationarity=False, 
                                                    enforce_invertibility=False)
                                    model_fit = model.fit(disp=False)

                                    result = {
                                        'order': order,
                                        'seasonal_order': seasonal_order,
                                        'model_fit': model_fit,
                                    }
                                    if len(train_data) <= 100:
                                        result['bic'] = model_fit.bic
                                        print(f"✅ SARIMA{order}x{seasonal_order} | BIC: {model_fit.bic:.2f}")
                                    else:
                                        result['aic'] = model_fit.aic
                                        print(f"✅ SARIMA{order}x{seasonal_order} | AIC: {model_fit.aic:.2f}")

                                    results.append(result)
                                except Exception as e:
                                    print(f"Failed for SARIMA{order}x{seasonal_order} - {e}")
                                    continue

    if not results:
        print("❌ No SARIMA models succeeded during tuning.")
        return None
    
    result_df = pd.DataFrame(results)
    if len(train_data) <= 100:
        top_models = result_df.nsmallest(top_n, 'bic')
    else:
        top_models = result_df.nsmallest(top_n, 'aic')
        
    best_score = float("inf")
    best_model = None
    
    for _, row in top_models.iterrows():
        try:
            order = row['order']
            seasonal_order = row['seasonal_order']
            model_fit = row['model_fit']
            forecast = model_fit.forecast(steps=len(test_data))
            rmse = root_mean_squared_error(test_data, forecast)
            print(f"🔍 Evaluating top model SARIMA{order}x{seasonal_order} | Test RMSE: {rmse:.4f}")
            if rmse < best_score:
                best_score = rmse
                if len(train_data) <= 100:
                    best_model = {
                        'order': row['order'],
                        'seasonal_order': row['seasonal_order'],
                        'bic': row['bic'],
                        'rmse': rmse,
                        'model_fit': row['model_fit']
                    }
                else:
                    best_model = {
                        'order': row['order'],
                        'seasonal_order': row['seasonal_order'],
                        'aic': row['aic'],
                        'rmse': rmse,
                        'model_fit': row['model_fit']
                    }
        except Exception as e:
            print(f"❌ Test RMSE calculation failed for SARIMA{row['order']}x{row['seasonal_order']} - {e}")
            
    if best_model:
        if len(train_data) <= 100:
            print(f"\n✅ Best SARIMA order: {best_model['order']}x{best_model['seasonal_order']} | RMSE: {best_model['rmse']:.4f} | BIC: {best_model['bic']}")
        else:
            print(f"\n✅ Best SARIMA order: {best_model['order']}x{best_model['seasonal_order']} | RMSE: {best_model['rmse']:.4f} | AIC: {best_model['aic']}")   
    else:
        print("❌ SARIMA hyperparameter tuning did not return a best model.")

    return best_model

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
    
def forecast_future(best_model, ts, target_col, train, test, last_actual_value, first_actual_value):
    """
    Forecast future values using a trained ARIMA or SARIMA model.
    
    Parameters:
    - final_fit: the fitted ARIMA/SARIMA model from tuning.
    - periods: number of future time steps to predict.
    - ts: your full time series DataFrame.
    - target_col: if multivariate, specify the target column name.
    
    Returns:
    - DataFrame with forecasted values.
    """

    periods = st.slider("Select the number of future periods to forecast:", min_value=1, max_value=365, value=7)
    
    # Inverse the differenced train and test data
    train_original = [first_actual_value + train[target_col].iloc[0]]  # Inverse differenced for the train data
    for i in range(1, len(train)):
        train_original.append(train_original[-1] + train[target_col].iloc[i])  # Add successive differenced values

    test_original = [train_original[-1] + test[target_col].iloc[0]]  # Inverse differenced for the test data
    for i in range(1, len(test)):
        test_original.append(test_original[-1] + test[target_col].iloc[i])  # Add successive differenced values

    
    # Forecast future periods using the model_fit stored in best_model
    model_fit = best_model.get('model_fit')
    if model_fit:
        forecast_diff = model_fit.forecast(steps=periods)
    
        if isinstance(forecast_diff, tuple):
            forecast = forecast[0]
            
        # Invert the differenced forecast back to the original scale
        forecast_original = [last_actual_value + forecast_diff[0]]  # The first forecasted value
        for i in range(1, len(forecast_diff)):
            forecast_original.append(forecast_original[-1] + forecast_diff[i])  # Add successive forecasted differences

        # Prepare index for new predictions
        last_date = ts.index[-1]
        freq = pd.infer_freq(ts.index)
        future_dates = pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train.index, y=train_original, mode='lines', name='Train Data', line=dict(color='blue', width=2)))
        fig.add_trace(go.Scatter(x=test.index, y=test_original, mode='lines', name='Test Data', line=dict(color='green', width=2)))
        fig.add_trace(go.Scatter(x=future_dates, y=forecast_original, mode='lines', name='Forecast', line=dict(color='red', width=2)))

        # Update layout for the plot
        fig.update_layout(
            title="Forecast vs Actuals",
            xaxis_title="Date",
            yaxis_title="Value",
            template="plotly_dark",
            hovermode="closest"  # This makes the hover info show for the closest point
        )
        st.plotly_chart(fig)

        return forecast_original
    else:
        st.error("No model fit found in the best model.")
        return None