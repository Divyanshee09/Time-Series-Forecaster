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
    
    y_train = train[[target_col]]
    y_test = test[[target_col]]

    order = (1, 1, 1)
    try:
        train_data = train if ts.shape[1] == 1 else y_train
        model = ARIMA(train_data, order=order)
        model_fit = model.fit()


    except Exception as e:
        print(f"❌ ARIMA({1},{1},{1}) failed: {e}")

    # Train RMSE
    train_pred = model_fit.fittedvalues
    train_rmse = np.sqrt(root_mean_squared_error(train_data, train_pred))
    print(f"📏 Train RMSE: {train_rmse:.4f}")

    # Test RMSE
    test_data = test if ts.shape[1] == 1 else y_test
    test_rmse = None
    if test_data is not None:
        try:
            forecast = model_fit.forecast(steps=len(test_data))
            test_rmse = np.sqrt(root_mean_squared_error(test_data, forecast))
            print(f"📏 Test RMSE: {test_rmse:.4f}")
        except Exception as e:
            print(f"❌ Test RMSE Calculation Failed: {e}")
            
    final_fit = model_fit

    return final_fit

def sarima_hyperparameter_tuning(ts, train, test, target_col, p_values, d_values, q_values, P_values, D_values, Q_values, S_values):
    train_data = train if ts.shape[1] == 1 else train[[target_col]]
    test_data = test if ts.shape[1] == 1 else test[[target_col]]

    warnings.filterwarnings("ignore")

    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 7)
    try:
        model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=False)

    except Exception as e:
        st.error(f"Failed for SARIMA{order}x{seasonal_order} - {e}")

    # Train RMSE
    train_pred = model_fit.fittedvalues
    train_rmse = np.sqrt(root_mean_squared_error(train_data, train_pred))
    print(f"📏 Train RMSE: {train_rmse:.4f}")

    # Test RMSE
    test_rmse = None
    if test_data is not None:
        try:
            forecast = model_fit.forecast(steps=len(test_data))
            test_rmse = np.sqrt(root_mean_squared_error(test_data, forecast))
            print(f"📏 Test RMSE: {test_rmse:.4f}")
        except Exception as e:
            print(f"❌ Test RMSE Calculation Failed: {e}")
    
    final_fit = model_fit

    return model_fit

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
        final_fit = sarima_hyperparameter_tuning(
            ts=ts,
            train=train,
            test=test,
            target_col=target_col,
            p_values=1,
            d_values=1,
            q_values=1,
            P_values=1,
            D_values=1,
            Q_values=1,
            S_values=7
        )
        return final_fit
    else:
        st.warning("⚠️ Weak or no seasonality detected. Proceeding with ARIMA tuning.")
        test_size = int(len(ts) * 0.2)
        train = ts.iloc[:-test_size]
        test = ts.iloc[-test_size:]
        final_fit = arima_hyperparameter_tuning(
            test=test,
            train=train,
            ts=ts,
            target_col=target_col,
            p_values=1,
            d_values=1,
            q_values=1
        )
        return final_fit
    
def forecast_future(final_fit, ts, target_col, train, test):
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

    periods = st.slider("Select the number of future periods to forecast:", min_value=1, max_value=365, value=15)
    
    # Forecast future periods
    forecast = final_fit.forecast(steps=periods)
    
    if isinstance(forecast, tuple):
        forecast = forecast[0]

    # Prepare index for new predictions
    last_date = ts.index[-1]
    freq = pd.infer_freq(ts.index)
    future_dates = pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train[target_col], mode='lines', name='Train Data', line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=test.index, y=test[target_col], mode='lines', name='Test Data', line=dict(color='green', width=2)))
    fig.add_trace(go.Scatter(x=future_dates, y=forecast, mode='lines', name='Forecast', line=dict(color='red', width=2)))

    # Update layout for the plot
    fig.update_layout(
        title="Forecast vs Actuals",
        xaxis_title="Date",
        yaxis_title="Value",
        template="plotly_dark",
        hovermode="closest"  # This makes the hover info show for the closest point
    )
    st.plotly_chart(fig)

    return forecast





























    # # Fit the model
    # model = SARIMAX(ts[target_col], 
    #                 order=(p, d, q), 
    #                 seasonal_order=(P, D, Q, s),
    #                 enforce_stationarity=False,
    #                 enforce_invertibility=False)
    # model_fit = model.fit(disp=False)
    
    # # Streamlit slider for user input
    # periods = st.slider("Select the number of future periods to forecast:", min_value=1, max_value=365, value=15)
    
    # # Make forecast
    # forecast = model_fit.forecast(steps=periods)
    
    # # Create forecast index (dates)
    # last_date = ts.index[-1]
    # forecast_index = pd.date_range(start=last_date + timedelta(days=1), periods=periods)
    # forecast = pd.Series(forecast, index=forecast_index)
    
    # # Ensure index is datetime
    # ts.index = pd.to_datetime(ts.index)

    # # Prediction
    # in_sample_pred = model_fit.get_prediction(start=ts.index[0], end=ts.index[-1])
    # in_sample_pred = in_sample_pred.predicted_mean

    # # Root Mean Squared Error
    # rmse = np.sqrt(root_mean_squared_error(ts[target_col].loc[in_sample_pred.index], in_sample_pred))

    
    # forecast_df = pd.DataFrame({
    #     'Forecast': forecast
    # }, index=forecast_index)
    
    
    # fig = go.Figure()

    # # Add actual data (blue solid line)
    # fig.add_trace(go.Scatter(x=ts.index, y=ts[target_col], mode='lines', name='Actual Data', line=dict(color='blue', width=2)))

    # # Add forecast data (red dashed line)
    # fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast, mode='lines', name='Forecast', line=dict(color='red', width=2)))

    # # Update layout for the plot
    # fig.update_layout(
    #     title="Forecast vs Actuals",
    #     xaxis_title="Date",
    #     yaxis_title="Value",
    #     template="plotly_dark",
    #     hovermode="closest"  # This makes the hover info show for the closest point
    # )

    # # Display the interactive plot in Streamlit
    # st.plotly_chart(fig)
    
    # return {
    #     'forecast': forecast,
    #     'model': model_fit,
    #     'rmse': rmse,
    #     'params': {'p': p, 'd': d, 'q': q, 'P': P, 'D': D, 'Q': Q, 's': s}
    # }
    
    
    
    
    # p_values=arima_params['p'],
    # d_values=arima_params['d'],
    # q_values=arima_params['q']