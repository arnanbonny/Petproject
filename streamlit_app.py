import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@st.cache_data
def load_data():
    df_nf = pd.read_csv("https://raw.githubusercontent.com/arnanbonny/Petproject/master/Clean_Dataset.csv")
    df = df_nf.drop(['flight', 'Unnamed: 0'], axis=1)
    return df

# Cache the entire model training pipeline
@st.cache_resource
def train_model(df):
    cat_cols = ['airline','source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city','class']
    num_cols = ['duration', 'days_left']
    
    # Encode categorical variables
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # Split features and target
    X = df_encoded.drop('price', axis=1)
    y = df_encoded['price']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])
    
    # Train model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test_scaled)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred)) 
    r2 = r2_score(y_test, y_pred)
    
    return rf_model, scaler, r2, mae, rmse, X_test_scaled, y_test, y_pred

# Load data and train model (cached, so only runs once)
with st.spinner('Loading data and training model...'):
    df = load_data()
    rf_model, scaler, r2, mae, rmse, X_test, y_test, y_pred = train_model(df)

# Display results
st.success('Model trained successfully!')

st.subheader('Model Performance Metrics')
col1, col2, col3 = st.columns(3)

with col1:
    st.metric('R² Score', f'{r2:.4f}')
with col2:
    st.metric('MAE', f'{mae:.2f}')
with col3:
    st.metric('RMSE', f'{rmse:.2f}')

st.write(r2)
