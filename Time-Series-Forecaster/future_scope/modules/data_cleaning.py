import streamlit as st
import pandas as pd
from scipy import stats
import numpy as np
from typing import Dict
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import PowerTransformer
import plotly.graph_objects as go
import warnings
from modules.utils import detect_outliers, detect_seasonality, detect_skewness, detect_trend

warnings.filterwarnings("ignore")

def making_data_continous(df, datetime_col, data_continous):
    
    if data_continous == "Daily":
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df = df.set_index(datetime_col)
        df = df.resample('D').asfreq()
        df = df.interpolate(method='time')
        df = df.reset_index()   
        df = df.infer_objects(copy=False)

    elif data_continous == "Weekly":
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df = df.set_index(datetime_col)
        df = df.resample('W').asfreq()
        df = df.interpolate(method='time')
        df = df.reset_index()
        df = df.infer_objects(copy=False)
        
    elif data_continous == "Monthly":
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df = df.set_index(datetime_col)
        df = df.resample('M').asfreq()
        df = df.interpolate(method='time')
        df = df.reset_index()
        df = df.infer_objects(copy=False)

    elif data_continous == "Quarterly":
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df = df.set_index(datetime_col)
        df = df.resample('Q').asfreq()
        df = df.interpolate(method='time')
        df = df.reset_index()
        df = df.infer_objects(copy=False)

    elif data_continous == "Half-Yearly":
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df = df.set_index(datetime_col)
        df = df.resample('2Q').asfreq()
        df = df.interpolate(method='time')
        df = df.reset_index()
        df = df.infer_objects(copy=False)
        
    elif data_continous == "Yearly":
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df = df.set_index(datetime_col)
        df = df.resample('Y').asfreq()
        df = df.interpolate(method='time')
        df = df.reset_index()
        df = df.infer_objects(copy=False)
    
    return df, data_continous
        
def handle_duplicates(df, target_col):

    cleaning_summary = {}
    duplicate_count_row = df.duplicated().sum()
    duplicate_count_column = df.T.duplicated().sum()
    
    st.write(f"Found {duplicate_count_row} duplicate rows and {duplicate_count_column} duplicate columns.")
    
    if duplicate_count_row > 0:
        df = df.drop_duplicates()
        cleaning_summary["Duplicates"] = f"Removed {duplicate_count_row} duplicate rows"

    if duplicate_count_column > 0:
        df = df.loc[:, ~df.columns.duplicated()]
        cleaning_summary["Duplicates"] = f"Removed {duplicate_count_column} duplicate columns"
    
    else:
        cleaning_summary["Duplicates"] = "No duplicates present"
           
    return df, cleaning_summary, target_col

def handling_null_values(df, cleaning_summary:Dict, datetime_col, target_col):
    
    null_values = df.isnull().sum()
    total_null = null_values.sum()

    st.write(f"Found {total_null} null values across all columns:")

    null_percentage = {
        column: (df[column].isnull().sum() / len(df) * 100).round(2) if len(df) > 0 and df[column].notnull().any() else None
        for column in df.columns
    }
    
    has_trend = detect_trend(df, target_col)
    has_seasonality = detect_seasonality(df, target_col)
    is_skewed = detect_skewness(df, target_col)
    has_outliers = detect_outliers(df, target_col)
        
    if null_percentage[target_col] is not None and null_percentage[target_col] < 5:
        #st.info(f"Null Percentage < 5 for {target_col}")
        df[target_col] = df[target_col].ffill()
        #st.info(f"Forward Fill in {target_col}")
            
    elif has_trend or has_seasonality:  
        df[target_col] = df[target_col].interpolate(method='time')
        #st.info(f"Trend/Seasonality, so linear interpolate in {target_col}")
            
    elif is_skewed or has_outliers:
        df[target_col] = df[target_col].fillna(df[target_col].median())
        #st.info(f"Skewed/Outliers, so median imputation in {target_col}")
            
    else:
        df[target_col] = df[target_col].fillna(df[target_col].mean())
        #st.info(f"Mean imputation in {target_col}")

    if  total_null > 0:
        cleaning_summary["Null Values"] = f"Handled {total_null} null values in the dataset"
        for column in df.columns:
            
                correlation = 0
                
                if pd.api.types.is_numeric_dtype(df[column]) and pd.api.types.is_numeric_dtype(df[target_col]):
                    # Fix: Check dtype correctly
                    if str(df[column].dtype) == 'float64' and str(df[target_col].dtype) in ['int64', 'float64']:
                        correlation = abs(df[column].corr(df[target_col]))
                        #st.info(f"There is correlation in {column}")
                    else:
                        correlation = 0
                        #st.info(f"No correlation in {column}")
                    
                if column in [target_col, datetime_col]:
                    continue

                if pd.api.types.is_numeric_dtype(df[column]):  # Only process numeric columns
                    has_trend = detect_trend(df, column)
                    has_seasonality = detect_seasonality(df, column)
                    is_skewed = detect_skewness(df, column)
                    has_outliers = detect_outliers(df, column)
        
                if null_percentage[column] is not None and null_percentage[column] > 50:
                    #st.info(f"Null Percentage > 50 in {column}")
                    if df[column].dtype == 'object':
                            #st.info(f"Null values > 50% and {column} data type is object so dropping the column")
                            df = df.drop(columns=[column])
                            
                    elif correlation > 0.5:
                        #st.info(f"Null Percentage > 50 and Correlation > 0.5 for {column}")
                            
                        if has_trend or has_seasonality:
                            df[column] = df[column].interpolate(method='time')
                            #st.info(f"Trend/Seasonality, so linear interpolate in {column}")
                            
                        elif is_skewed or has_outliers:
                            df[column] = df[column].fillna(df[column].median())
                            #st.info(f"Skewed/Outliers, so median imputation in {column}")
                            
                        else:
                            df[column] = df[column].fillna(df[column].mean())
                            #st.info(f"Correlation < 0.5 in {column} so dropping the null values")
                    else:
                        df = df.drop(columns=[column])
                else:
                    #st.info(f"Null Percentage < 50 {column}")
                    if df[column].dtype == 'object':
                        df[column] = df[column].fillna(df[column].mode()[0])
                        #st.info(f"Object Type, so mode imputation in {column}")
                        
                    elif null_percentage[column] is not None and null_percentage[column] < 5:
                        #st.info(f"Null Percentage < 5 for {column}")
                        df[column] = df[column].ffill()
                        #st.info(f"Forward Fill in {column}")

                    elif has_trend or has_seasonality:  
                        df[column] = df[column].interpolate(method='linear')
                        #st.info(f"Trend/Seasonality, so linear interpolate in {column}")
                            
                    elif is_skewed or has_outliers:
                        df[column] = df[column].fillna(df[column].median())
                        #st.info(f"Skewed/Outliers, so median imputation in {column}")
                            
                    else:
                        df[column] = df[column].fillna(df[column].mean())
                        #st.info(f"Mean imputation in {column}")
    else:
        st.success("No null values found in the dataset!")
        cleaning_summary["Null Values"] = "No null values found"

    #cleaning_summary["Data Size"] = f"{df.shape}"
    return df, cleaning_summary, target_col

def outlier_detection(df, cleaning_summary, datetime_col, target_col):
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != datetime_col]
    
    outlier_indices = []
    
    for column in df.columns:
        if column == datetime_col:
            continue
        
        if not pd.api.types.is_numeric_dtype(df[column]):
            continue
        
        has_outliers = detect_outliers(df, column)
        #st.info(f"{column} has outliers: {has_outliers}")
        
        if has_outliers != False:
            _, p_value = stats.shapiro(df[column]) if len(df[column]) <= 5000 else (0, 0)    # Normality Test
            if df[column].nunique() > 1:  
                skewness = stats.skew(df[column] + 1e-10)  
            else:  
                skewness = 0
                
            if len(df[column]) < 10000:
                if p_value > 0.05 and abs(skewness) < 0.5:
                    z_scores = np.abs(stats.zscore(df[column]))
                    outlier_indices = np.where(z_scores > 3)[0]
                    #st.info(f"Outliers detected using Z-Score in {column}")

                else:
                    q1 = np.percentile(df[column], 25)
                    q3 = np.percentile(df[column], 75)
                    iqr = q3 - q1
                    lower_bound = q1 - (1.5 * iqr)
                    upper_bound = q3 + (1.5 * iqr)
                    outlier_indices = np.where((df[column] < lower_bound) | (df[column] > upper_bound))[0]
                    #st.info(f"Outliers detected using IQR in {column}")

            else:
                iso_forest = IsolationForest(contamination=0.05, random_state=42)
                predictions = iso_forest.fit_predict(df[column].values.reshape(-1, 1))
                outlier_indices = np.where(predictions == -1)[0]
                #st.info(f"Outliers detected using Iso-Forest in {column}")
                
    numeric_df = df.select_dtypes(include='number')

    # Outlier Detection - Before Treatment
    if not numeric_df.empty:
        st.subheader("📉 Outlier Detection (Before Treatment)")
        fig = go.Figure()
        for col in numeric_df.columns:
            fig.add_trace(go.Box(
                x=numeric_df[col],
                name=col,
                boxpoints='outliers',
                marker_color='indianred',
                orientation='h'
            ))

        fig.update_layout(
            title="Outliers Detection",
            xaxis_title="Value",
            yaxis_title="Feature",
            template="plotly_white",
            height=50 + len(numeric_df.columns)*40,
            margin=dict(l=80, r=40, t=40, b=40),
            boxmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    return df, outlier_indices, cleaning_summary, target_col

def handle_outliers(df, cleaning_summary:Dict, datetime_col, outlier_indices, target_col):

    outlier_percentage = {
            column: (len(outlier_indices) / len(df) * 100) if len(df) > 0 else None
            for column in df.columns
        }

    for column in df.columns:
            if column == datetime_col:
                continue
            
            if not pd.api.types.is_numeric_dtype(df[column]):
                continue
            
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            _, p_value = stats.shapiro(df[column]) if len(df) <= 5000 else (0, 0)  # Normality Test
            if df[column].nunique() > 1:  
                skewness = stats.skew(df[column] + 1e-10)  
            else:  
                skewness = 0
            #st.info(f"Skewness for {column}: {skewness}")
            
            cleaning_summary["Outliers"] = f"{len(outlier_indices)} outliers treated"
            
            if outlier_percentage[column] is not None and outlier_percentage[column] < 5:
                df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
                #st.info(f"Dropped {len(outlier_indices)} outliers from {column} (Safe Removal)")

            else:
                if skewness > 1.0:
                    if skewness > 1.5 and df[column].min() > 0:
                        # Box-Cox (only for positive values)
                        df[column], _ = stats.boxcox(df[column])
                        #st.info(f"Applied Box-Cox transformation on {column}")

                    elif 1.0 < skewness <= 1.5 and df[column].min() >= 0:
                        # Log1p if moderately skewed & non-negative
                        df[column] = np.log1p(df[column])
                        #st.info(f"Applied Log transformation on {column}")

                    elif skewness < -1 or skewness > 1:
                        # Yeo-Johnson if either skewed and Box-Cox not possible
                        pt = PowerTransformer(method='yeo-johnson')
                        df[column] = pt.fit_transform(df[[column]])
                        #st.info(f"Applied Yeo-Johnson transformation on {column}")
                        
                elif len(df[column]) < 5000:
                    # Small dataset → Replace outliers with median
                    median_value = df[column].median()
                    df[column] = np.where((df[column] < lower_bound) | (df[column] > upper_bound), 
                                            median_value, df[column])
                    #st.info(f"Replaced outliers in {column} with median")

                elif 5000 <= len(df[column]) < 10000:
                    # If data is normally distributed → Use Z-Score method
                    if p_value > 0.05 and abs(skewness) < 0.5:
                        z_scores = np.abs(stats.zscore(df[column]))
                        df = df[z_scores < 3]  # Remove extreme Z-score outliers
                        #st.info(f"Removed Z-score outliers in {column}")

                    else:
                        # Skewed Data → Use Winsorization (Capping)
                        df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
                        df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
                        #st.info(f"Capped outliers in {column} within IQR bounds")

                else:
                    # Large dataset → Use Isolation Forest
                    iso_forest = IsolationForest(contamination=0.05, random_state=42)
                    predictions = iso_forest.fit_predict(df[[column]])  # Expecting 2D input
                    df = df.copy()
                    df.loc[:, 'outlier_flag'] = predictions
                    df = df[df['outlier_flag'] == 1]  # Keep only non-outliers
                    df.drop(columns=['outlier_flag'], inplace=True)
                    #st.info(f"Removed outliers in {column} using Isolation Forest")
    
    numeric_df = df.select_dtypes(include='number')

    if not numeric_df.empty:
        st.subheader("✅ Outlier Treated (After Cleaning)")
        fig = go.Figure()
        for col in numeric_df.columns:
            fig.add_trace(go.Box(
                x=numeric_df[col],
                name=col,
                boxpoints='outliers',
                marker_color='seagreen',
                orientation='h'
            ))

        fig.update_layout(
            title="Outliers After Treatment",
            xaxis_title="Value",
            yaxis_title="Feature",
            template="plotly_white",
            height=50 + len(numeric_df.columns)*40,
            margin=dict(l=80, r=40, t=40, b=40),
            boxmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)

    return df, cleaning_summary, target_col

def fixing_data_types(data, df, cleaning_summary, target_col):
    
    # ---- Data type conversion ----
    st.write("Current data types:")
    st.write(df.dtypes)
    
    # Select columns for type conversion
    columns_for_conversion = st.multiselect(
        "Select columns to convert data type:",
        options=df.columns.tolist()
    )
    
    if columns_for_conversion:
        for col in columns_for_conversion:
            current_type = str(df[col].dtype)
            new_type = st.selectbox(
                f"Convert {col} from {current_type} to:",
                options=["float64", "int64", "string", "category", "datetime", "boolean"],
                key=f"convert_{col}"
            )
            
            try:
                if new_type == "float64":
                    df[col] = df[col].astype(float)
                elif new_type == "int64":
                    # For int conversion, handle nulls first
                    if df[col].isnull().any():
                        st.warning(f"Column {col} contains NaN values which can't be converted to int directly.")
                        fill_value = st.number_input(f"Fill NaN values in {col} with:", value=0)
                        df[col] = df[col].fillna(fill_value).astype(int)
                    else:
                        df[col] = df[col].astype(int)
                elif new_type == "string":
                    df[col] = df[col].astype(str)
                elif new_type == "category":
                    df[col] = df[col].astype('category')
                elif new_type == "datetime":
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                elif new_type == "boolean":
                    df[col] = df[col].astype(bool)
                
                if "data_types" not in cleaning_summary:
                    cleaning_summary["data_types"] = []
                cleaning_summary["data_types"].append(f"Converted {col} to {new_type}")
                
            except Exception as e:
                st.error(f"Error converting {col} to {new_type}: {str(e)}")
    
    return df, cleaning_summary, target_col