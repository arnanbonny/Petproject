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
st.divider()
st.header('✈️ Predict Flight Price')
st.write('Enter flight details below to get a price prediction:')

# Create input form
with st.form('prediction_form'):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        airline = st.selectbox('Airline', options=sorted(df['airline'].unique()))
        source_city = st.selectbox('Source City', options=sorted(df['source_city'].unique()))
        departure_time = st.selectbox('Departure Time', options=sorted(df['departure_time'].unique()))
    
    with col2:
        destination_city = st.selectbox('Destination City', options=sorted(df['destination_city'].unique()))
        arrival_time = st.selectbox('Arrival Time', options=sorted(df['arrival_time'].unique()))
        stops = st.selectbox('Number of Stops', options=sorted(df['stops'].unique()))
    
    with col3:
        flight_class = st.selectbox('Class', options=sorted(df['class'].unique()))
        duration = st.slider('Duration (hours)', 
                            min_value=float(df['duration'].min()), 
                            max_value=float(df['duration'].max()), 
                            value=float(df['duration'].mean()),
                            step=0.5)
        days_left = st.slider('Days Left Until Departure', 
                             min_value=int(df['days_left'].min()), 
                             max_value=int(df['days_left'].max()), 
                             value=int(df['days_left'].mean()))
    
    submit_button = st.form_submit_button('🔮 Predict Price', use_container_width=True)

if submit_button:
    # Create input dataframe
    input_data = pd.DataFrame({
        'airline': [airline],
        'source_city': [source_city],
        'departure_time': [departure_time],
        'stops': [stops],
        'arrival_time': [arrival_time],
        'destination_city': [destination_city],
        'class': [flight_class],
        'duration': [duration],
        'days_left': [days_left]
    })
    
    # Encode categorical variables (same as training)
    cat_cols = ['airline','source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city','class']
    input_encoded = pd.get_dummies(input_data, columns=cat_cols, drop_first=True)
    
    # Align columns with training data
    # Get all columns from training data
    training_cols = X_train.columns
    
    # Add missing columns with 0
    for col in training_cols:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    
    # Reorder columns to match training data
    input_encoded = input_encoded[training_cols]
    
    # Scale numerical features
    num_cols = ['duration', 'days_left']
    input_scaled = input_encoded.copy()
    input_scaled[num_cols] = scaler.transform(input_encoded[num_cols])
    
    # Make prediction
    predicted_price = rf_model.predict(input_scaled)[0]
    
    # Display prediction with styling
    st.success('### 💰 Predicted Flight Price')
    st.markdown(f"<h1 style='text-align: center; color: #1f77b4;'>₹ {predicted_price:,.2f}</h1>", 
               unsafe_allow_html=True)
    
    # Show input summary
    with st.expander('📝 Input Summary'):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Route:** {source_city} → {destination_city}")
            st.write(f"**Airline:** {airline}")
            st.write(f"**Class:** {flight_class}")
            st.write(f"**Stops:** {stops}")
        with col2:
            st.write(f"**Departure:** {departure_time}")
            st.write(f"**Arrival:** {arrival_time}")
            st.write(f"**Duration:** {duration} hours")
            st.write(f"**Days Left:** {days_left} days")


