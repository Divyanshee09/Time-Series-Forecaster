import pandas as pd
import streamlit as st
import io
import warnings
from modules.file_loader import load_time_series_data
from modules.data_cleaning import making_data_continous, handle_duplicates, handling_null_values, outlier_detection, handle_outliers, fixing_data_types
from modules.data_preprocessing import index_set, is_white_noise
from modules.data_preprocessing import stationarity
from modules.decomposition import trend_seasonality_analysis
from modules.train_test_split import test_train_split, scaling
from modules.model_tuning import auto_select_arima_or_sarima
from modules.model_tuning import forecast_future

warnings.filterwarnings("ignore")

visualization_tab, duplicates_tab, null_tab, outliers_tab, dtypes_tab, decomposition_tab = st.tabs([
    "Forecasted Results", "Duplicates", "Null Values", "Outliers", "Fix Data Types", "Time Series Decomposition"
])

def integrate_with_time_series_app():
    # Step 1: Load the data first
    data,filetype, datetime_col = load_time_series_data()
    
    # Step 2: If data is loaded, proceed to cleaning
    if data is not None:
        
        if datetime_col is not None:
            st.info(f"Datetime column '{datetime_col}' detected.")
        else:
            # If no datetime column was found, ask user to select one
            datetime_cols = [col for col in data.columns if 'date' in col.lower() or 'time' in col.lower()]
            
            if not datetime_cols:
                datetime_cols = data.columns.tolist()
            
            datetime_col = st.selectbox(
                "Select the datetime column:", 
                options=datetime_cols
            )
    
        df = data.copy()
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != datetime_col]

        # Selecting target variable
        target_col = None  
        if numeric_cols:
            target_col = st.selectbox(
                        "Select Target Variable:",
                        options=numeric_cols,
                        index=None
                        )

        data_continous = st.selectbox(
            "Select Data Frequency:",
            options=["Select",
                    "Daily", 
                    "Weekly", 
                    "Monthly", 
                    "Quarterly", 
                    "Half-Yearly", 
                    "Yearly"],
            index = None
        )
        
        # Initialize variables
        entity_column = None
        entity = None
        
        if df[datetime_col].duplicated().any():
            entity_column = st.selectbox(
                "Select Entity Column:",
                options=list(df.columns),
                index = None
            )
            
            # entity = "Select"
            if entity_column is not None:
                entity = st.selectbox(
                    f"Select the Entity Value from {entity_column}:",
                    options=list(map(str, df[entity_column].dropna().unique())),
                    index = None
                )
        # Check if all selections are made before proceeding
        if (target_col is not None
            and data_continous != None
            and (not df[datetime_col].duplicated().any() or 
                (entity_column is not None and entity is not None))):
            st.success("All selections are valid. Proceeding with analysis...")

            # Filter dataframe if entity is selected
            if entity_column is not None and entity is not None:
                df = df[df[entity_column] == entity]
            
            if df[datetime_col].duplicated().any():
                # Aggregate target column by date
                df = df.groupby(datetime_col)[target_col].mean().reset_index()
                st.info(f"Aggregated multiple {target_col} for {entity_column} : '{entity}' using mean.")
                    
            df, data_continous = making_data_continous(df, datetime_col, data_continous)
            with duplicates_tab:
                df, cleaning_summary, target_col = handle_duplicates(df, target_col)
            with null_tab:
                df, cleaning_summary, target_col = handling_null_values(df, cleaning_summary, datetime_col, target_col)
            with outliers_tab:
                df, outlier_indices, cleaning_summary, target_col = outlier_detection(df, cleaning_summary, datetime_col, target_col)
                if len(outlier_indices) > 0:
                    df, cleaning_summary, target_col = handle_outliers(df, cleaning_summary, datetime_col, outlier_indices, target_col)
                else:
                    st.success("No outliers found in the dataset!")
                    cleaning_summary["Outliers"] = "No outliers present"
            with dtypes_tab:
                df, cleaning_summary, target_col = fixing_data_types(data, df, cleaning_summary, target_col)
            
            cleaning_summary["Data Shape"] = f"Original data shape: {data.shape[0]} rows, {data.shape[1]} columns | Cleaned data shape: {df.shape[0]} rows, {df.shape[1]} columns"

            # Display Cleaning Summary
            if cleaning_summary:
                st.subheader("Data Cleaning Summary")
                for step, detail in cleaning_summary.items():
                    st.write(f"- {detail}")
            else:
                st.info("No cleaning operations were performed.")
                
            # Preview the cleaned data
            st.subheader("Cleaned Data Preview")

            if datetime_col is not None and datetime_col in df.columns:
                st.dataframe(df)
            
            ts, datetime_col = index_set(df, datetime_col, data_continous)
            if is_white_noise(ts, target_col, alpha=0.05):
                    st.error("🚫 The series appears to be white noise. Forecasting/analysis is not recommended.")
            else:
                with decomposition_tab:
                    ts, seasonal_period = trend_seasonality_analysis(ts, target_col)
                    ts[target_col], last_actual_value, first_actual_value = stationarity(ts, target_col, max_diff=3)
                    ts = ts.dropna(subset=[target_col])
                
                # Test-train split
                train, test = test_train_split(ts, target_col, entity_column)
                
                # if ts.shape[1]==1 or all(ts[col].dtype == 'object' for col in ts.columns if col != target_col):
                #     train_scaled, test_scaled = scaling(ts, train, test, target_col)
                # else:
                #     y_train, y_test, x_train, x_test, x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled = scaling(ts, train, test, target_col)
                
                arima_params = {'p': range(0, 3), 'd': range(0, 2), 'q': range(0, 3)}
                sarima_params = {
                    'p': range(0, 3), 'd': range(0, 2), 'q': range(0, 3),
                    'P': range(0, 3), 'D': range(0, 2), 'Q': range(0, 3), 'S': [7, 14]
                }
                
                with visualization_tab:
                    # result = sarima_hyperparameter_tuning(ts, train, test, target_col, **sarima_params)
                    best_model = auto_select_arima_or_sarima(ts, train, test, target_col, arima_params, sarima_params, seasonal_period)
                    forecast = forecast_future(best_model, ts, target_col, train, test, last_actual_value, first_actual_value)
                        
            if df is not None:
                st.session_state['cleaned_data'] = df
                st.session_state['datetime_col'] = datetime_col
                
        else:
            st.warning("Please complete all selections above to continue.")
                
            # Convert DataFrame to CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        # Create download button
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="data.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    integrate_with_time_series_app()


# import pandas as pd
# import streamlit as st
# import io
# import warnings
# import plotly.express as px
# from modules.file_loader import load_time_series_data
# from modules.data_cleaning import making_data_continous, handle_duplicates, handling_null_values, outlier_detection, handle_outliers, fixing_data_types
# from modules.data_preprocessing import index_set, is_white_noise, stationarity
# from modules.decomposition import trend_seasonality_analysis
# from modules.train_test_split import test_train_split, scaling
# from modules.model_tuning import auto_select_arima_or_sarima, forecast_future

# def get_session_state(key, default=None):
#     return st.session_state[key] if key in st.session_state else default

# warnings.filterwarnings("ignore")

# st.set_page_config(page_title="Time Series Forecaster", layout="wide", initial_sidebar_state="expanded")

# st.markdown("""
#     <style>
#         .main {background-color: #0e1117; color: white;}
#         .stApp {background-color: #0e1117; color: white;}
#     </style>
# """, unsafe_allow_html=True)

# st.title("📈 Time Series Forecasting App")
# st.sidebar.title("🔧 Navigation")

# section = st.sidebar.radio("Go to", [
#     "Upload & Configure",
#     "Data Cleaning",
#     "Decomposition & Stationarity",
#     "Modeling & Forecasting",
#     "Download Forecast"
# ])

# # Initialize session state keys
# for key in ['cleaned_data', 'datetime_col', 'target_col', 'data_continous', 'ts', 'seasonal_period', 'last_val', 'forecast']:
#     if key not in st.session_state:
#         st.session_state[key] = None

# if section == "Upload & Configure":
#     st.header("🔍 Step 1: Upload and Setup")
#     df, cleaning_summary, datetime_col = load_time_series_data()

#     if df is not None:
#         st.success("Data uploaded successfully!")

#         if datetime_col is not None:
#             st.info(f"Datetime column '{datetime_col}' detected.")
#         else:
#             # If no datetime column was found, ask user to select one
#             datetime_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            
#             if not datetime_cols:
#                 datetime_cols = df.columns.tolist()
            
#             datetime_col = st.selectbox(
#                 "Select the datetime column:", 
#                 options=datetime_cols
#             )

#         numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
#         numeric_cols = [col for col in numeric_cols if col != datetime_col]

#         # Selecting target variable
#         target_col = None  
#         if numeric_cols:
#             target_col = st.selectbox(
#                         "Select Target Variable:",
#                         options=numeric_cols,
#                         index=None
#                         )

#         data_continous = st.selectbox(
#             "Select Data Frequency:",
#             options=["Select",
#                     "Daily", 
#                     "Weekly", 
#                     "Monthly", 
#                     "Quarterly", 
#                     "Half-Yearly", 
#                     "Yearly"],
#             index = None
#         )
        
#         #Initialize variables
#         entity_column = None
#         entity = None
        
#         if df[datetime_col].duplicated().any():
#             entity_column = st.selectbox(
#                 "Select Entity Column:",
#                 options=list(df.columns),
#                 index = None
#             )
            
#             # entity = "Select"
#             if entity_column is not None:
#                 entity = st.selectbox(
#                     f"Select the Entity Value from {entity_column}:",
#                     options=list(map(str, df[entity_column].dropna().unique())),
#                     index = None
#                 )
                
#             # Filter dataframe if entity is selected
#             if entity_column is not None and entity is not None:
#                 df = df[df[entity_column] == entity]
                
#             if df[datetime_col].duplicated().any():
#                 # Aggregate target column by date
#                 df = df.groupby(datetime_col)[target_col].mean().reset_index()
#                 st.info(f"Aggregated multiple {target_col} for {entity_column} : '{entity}' using mean.")
                
#         # Check if all selections are made before proceeding
#         if (target_col is not None
#             and data_continous != None
#             and (not df[datetime_col].duplicated().any() or 
#                 (entity_column is not None and entity is not None))):
#             st.success("All selections are valid. Proceeding with cleaning...")
#             st.session_state['cleaned_data'] = df
#             st.session_state['datetime_col'] = datetime_col
#             st.session_state['target_col'] = target_col
#             st.session_state['data_continous'] = data_continous

#             #st.success("Configuration complete. Proceed to cleaning.")

# if section == "Data Cleaning" and st.session_state['cleaned_data'] is not None:
#     st.header("🧼 Step 2: Data Cleaning")
#     df = st.session_state['cleaned_data']
#     datetime_col = st.session_state['datetime_col']
#     target_col = st.session_state['target_col']
#     data_continous = st.session_state['data_continous']
    
#     if datetime_col and target_col and data_continous:
#         df, data_continous = making_data_continous(df, datetime_col, data_continous)

#     with st.expander("Handle Duplicates"):
#         df, _, _ = handle_duplicates(df, target_col)
#     with st.expander("Handle Null Values"):
#         df, _, _ = handling_null_values(df, {}, datetime_col, target_col)
#     with st.expander("Outlier Detection"):
#         df, outliers, _, _ = outlier_detection(df, {}, datetime_col, target_col)
#         if outliers:
#             df, _, _ = handle_outliers(df, {}, datetime_col, outliers, target_col)
#     with st.expander("Fix Data Types"):
#         df, _, _ = fixing_data_types(df, df, {}, target_col)

#     st.session_state['cleaned_data'] = df
#     st.session_state['data_continous'] = data_continous
#     st.success("Cleaning complete!")
#     st.dataframe(df)


# df = st.session_state['cleaned_data']
# datetime_col = st.session_state['datetime_col']
# data_continous = st.session_state['data_continous']
# ts, datetime_col = index_set(df, datetime_col, data_continous)
# if is_white_noise(ts, target_col, alpha=0.05):
#     st.error("Series appears to be white noise. Forecasting not recommended.")
# else:
#     if section == "Decomposition & Stationarity" and st.session_state['cleaned_data'] is not None:
#         st.header("🔍 Step 3: Decomposition & Stationarity")
#         df = st.session_state['cleaned_data']
#         datetime_col = st.session_state['datetime_col']
#         target_col = st.session_state['target_col']
#         frequency = st.session_state['frequency']


#         ts, seasonal_period = trend_seasonality_analysis(ts, target_col)
#         ts[target_col], last_actual_value, first_actual_value = stationarity(ts, target_col, max_diff=3)
#         ts.dropna(subset=[target_col], inplace=True)
#         st.session_state['ts'] = ts
#         st.session_state['seasonal_period'] = seasonal_period
#         st.session_state['last_val'] = last_actual_value

#         if section == "Modeling & Forecasting" and 'ts' in st.session_state:
#             st.header("🧠 Step 4: Model Training & Forecasting")
#             ts = st.session_state['ts']
#             target_col = st.session_state['target_col']
#             seasonal_period = st.session_state['seasonal_period']
#             last_val = st.session_state['last_val']

#             with st.spinner("Training model..."):
#                 train, test = test_train_split(ts, target_col)
#                 if ts.shape[1] == 1 or all(ts[col].dtype == 'object' for col in ts.columns if col != target_col):
#                     train_scaled, test_scaled = scaling(ts, train, test, target_col)
#                 else:
#                     y_train, y_test, x_train, x_test, x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled = scaling(ts, train, test, target_col)

#                 arima_params = {'p': range(0, 3), 'd': range(0, 2), 'q': range(0, 3)}
#                 sarima_params = {
#                     'p': range(0, 3), 'd': range(0, 2), 'q': range(0, 3),
#                     'P': range(0, 3), 'D': range(0, 2), 'Q': range(0, 3), 'S': [7, 12]
#                 }

#                 best_model = auto_select_arima_or_sarima(ts, train, test, target_col, arima_params, sarima_params, seasonal_period)
#                 forecast_df = forecast_future(best_model, ts, target_col, train, test, last_val, st.session_state['ts'].iloc[0][target_col])
#                 st.session_state['forecast'] = forecast_df

#             # st.success("Forecasting complete!")
#             # fig = px.line(forecast_df, x=forecast_df.index, y=['actual', 'forecast'], title="Forecast vs Actual")
#             # st.plotly_chart(fig, use_container_width=True)

#             st.metric("RMSE", f"{forecast_df['rmse'].iloc[0]:.2f}")

# if section == "Download Forecast" and 'forecast' in st.session_state:
#     st.header("📥 Step 5: Download Forecast")
#     forecast_df = st.session_state['forecast']
#     csv = forecast_df.to_csv(index=True)
#     st.download_button("📁 Download Forecast CSV", csv, "forecast_output.csv", "text/csv")
