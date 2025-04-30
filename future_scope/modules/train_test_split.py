import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")

def test_train_split(ts, target_col, entity_column):
    test_size_percentage = 0.2  # for example, 20%

    # Calculate split index
    split_index = int(len(ts) * (1 - test_size_percentage))

    # Perform train-test split
    if entity_column == 'emp_id':
        train = ts.iloc[:-7]
        test = ts.iloc[-7:]
    else:
        train = ts.iloc[:split_index]
        test = ts.iloc[split_index:]
    
    print(f"Train size: {len(train)}")
    print(f"Test size: {len(test)}")

    return train, test

def scaling(ts, train, test, target_col):
    
    # Create scaler object
    scaler = MinMaxScaler()
    
    if ts.shape[1] == 1 or all(ts[col].dtype == 'object' for col in ts.columns if col != target_col):
        train_scaled = scaler.fit_transform(train[[target_col]])
        test_scaled = scaler.transform(test[[target_col]])

        print("train_scaled Min:", train_scaled.min(), "Max:", train_scaled.max())
        print("test_scaled Min:", test_scaled.min(), "Max:", test_scaled.max())
        
        return train_scaled, test_scaled
        
    else:
        # Fit only on training features (excluding target)
        x_train = train.drop(columns=[target_col])
        x_test = test.drop(columns=[target_col])
        
        # Drop object-type columns
        x_train_numeric = x_train.select_dtypes(exclude=['object', 'datetime'])
        x_test_numeric = x_test.select_dtypes(exclude=['object', 'datetime'])

        # Fit the scaler on train, transform both train and test
        x_train_scaled = scaler.fit_transform(x_train_numeric)
        x_test_scaled = scaler.transform(x_test_numeric)

        # If you also want to scale the target
        target_scaler = MinMaxScaler()

        y_train = train[[target_col]]
        y_test = test[[target_col]]
        
        y_train = y_train.dropna()
        y_test = y_test.dropna()
        
        y_train_scaled = target_scaler.fit_transform(y_train)
        y_test_scaled = target_scaler.transform(y_test)
        
        print("X_train_scaled Min:", x_train_scaled.min(), "Max:", x_train_scaled.max())
        print("X_test_scaled Min:", x_test_scaled.min(), "Max:", x_test_scaled.max())

        print("y_train_scaled Min:", y_train_scaled.min(), "Max:", y_train_scaled.max())
        print("y_test_scaled Min:", y_test_scaled.min(), "Max:", y_test_scaled.max())
        
        print("Dropping columns:", x_train.select_dtypes(include=['object']).columns.tolist())

        return y_train, y_test, x_train, x_test, x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled