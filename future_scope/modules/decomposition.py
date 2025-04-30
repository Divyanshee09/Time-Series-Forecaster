import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose, STL

def trend_seasonality_analysis(ts, target_col):
    st.subheader("Original Time Series")
    fig1 = px.line(ts, y=target_col, title='Time Series Overview')
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Seasonal Decomposition (Additive)")
    decomposition = seasonal_decompose(ts[target_col], model='additive', period=12)
    additive_df = pd.DataFrame({
        "Trend": decomposition.trend,
        "Seasonality": decomposition.seasonal,
        "Residual": decomposition.resid
    })

    def make_decomposition_plot(index, original, components_df, title):
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                            subplot_titles=("Original", "Trend", "Seasonality", "Residual"))

        fig.add_trace(go.Scatter(x=index, y=original, mode='lines', name='Original'), row=1, col=1)
        fig.add_trace(go.Scatter(x=index, y=components_df["Trend"], mode='lines', name='Trend'), row=2, col=1)
        fig.add_trace(go.Scatter(x=index, y=components_df["Seasonality"], mode='lines', name='Seasonality'), row=3, col=1)
        fig.add_trace(go.Scatter(x=index, y=components_df["Residual"], mode='lines', name='Residual'), row=4, col=1)

        fig.update_layout(height=800, title_text=title, showlegend=False)
        return fig

    fig2 = make_decomposition_plot(ts.index, ts[target_col], additive_df, "Seasonal Decomposition (Additive)")
    st.plotly_chart(fig2, use_container_width=True)

    if not (ts[target_col] <= 0).any():
        st.subheader("Seasonal Decomposition (Multiplicative)")
        decomposition_mul = seasonal_decompose(ts[target_col], model='multiplicative', period=12)
        multiplicative_df = pd.DataFrame({
            "Trend": decomposition_mul.trend,
            "Seasonality": decomposition_mul.seasonal,
            "Residual": decomposition_mul.resid
        })
        fig3 = make_decomposition_plot(ts.index, ts[target_col], multiplicative_df, "Seasonal Decomposition (Multiplicative)")
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("STL Decomposition (Seasonal-Trend decomposition using LOESS)")

    seasonal_options = [i for i in range(7, 366, 2)]  # Odd numbers from 7 to 365
    seasonal_period = st.selectbox("Select Seasonal Period:", options=seasonal_options, index=seasonal_options.index(365))

    stl = STL(ts[target_col], seasonal=seasonal_period)
    result = stl.fit()

    def make_stl_plot(index, result):
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                            subplot_titles=("Observed", "Trend", "Seasonal", "Residual"))

        fig.add_trace(go.Scatter(x=index, y=result.observed, mode='lines', name='Observed'), row=1, col=1)
        fig.add_trace(go.Scatter(x=index, y=result.trend, mode='lines', name='Trend'), row=2, col=1)
        fig.add_trace(go.Scatter(x=index, y=result.seasonal, mode='lines', name='Seasonal'), row=3, col=1)
        fig.add_trace(go.Scatter(x=index, y=result.resid, mode='lines', name='Residual'), row=4, col=1)

        fig.update_layout(height=800, title_text="STL Decomposition", showlegend=False)
        return fig

    st.plotly_chart(make_stl_plot(ts.index, result), use_container_width=True)

    components_df = pd.concat([
        result.trend.rename("Trend"),
        result.seasonal.rename("Seasonality"),
        result.resid.rename("Residual")
    ], axis=1)

    st.subheader("STL Components (Line Chart)")
    st.line_chart(components_df)

    return ts, seasonal_period
















# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import plotly.express as px
# import plotly.graph_objects as go
# from statsmodels.tsa.stattools import adfuller
# from statsmodels.tsa.seasonal import seasonal_decompose
# from statsmodels.tsa.seasonal import STL
# import warnings

# warnings.filterwarnings("ignore")

# def trend_seasonality_analysis(ts, target_col):
    
#     st.subheader("Original Time Series")
#     fig1, ax1 =  plt.subplots(figsize=(14, 6))
#     ax1.plot(ts[target_col], label='Time Series')
#     ax1.set_title('Time Series Overview')
#     ax1.legend()
#     st.pyplot(fig1)
    
#     st.subheader("Seasonal Decomposition (Additive)")
#     decomposition = seasonal_decompose(ts[target_col], model='additive')
#     fig2 = decomposition.plot()
#     fig2.set_size_inches(14, 8)
#     st.pyplot(fig2)
    
#     if not (ts[target_col] <= 0).any():
#         st.subheader("Seasonal Decomposition (Multiplicative)")
#         decomposition = seasonal_decompose(ts[target_col], model='multiplicative')
#         fig3 = decomposition.plot()
#         fig3.set_size_inches(14, 8)
#         st.pyplot(fig3)
    
#     # STL Decomposition
#     st.subheader("STL Decomposition (Seasonal-Trend decomposition using LOESS)")

#     # Only allow odd seasonal periods >= 7
#     seasonal_options = [i for i in range(7, 366, 2)]  # Odd numbers from 7 to 365
#     seasonal_period = st.selectbox("Select Seasonal Period:", options=seasonal_options, index=seasonal_options.index(365))

#     stl = STL(ts[target_col], seasonal=seasonal_period)
#     result = stl.fit()

#     # Plot STL result
#     fig4 = result.plot()        
#     fig4.set_size_inches(14, 8)
#     st.pyplot(fig4)

#     st.subheader("STL Components")
#     components_df = pd.concat([
#         result.trend.rename("Trend"),
#         result.seasonal.rename("Seasonality"),
#         result.resid.rename("Residual")
#     ], axis=1)

#     st.line_chart(components_df)
#     return ts, seasonal_period