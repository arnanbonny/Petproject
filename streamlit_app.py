import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configure page - this should be first
st.set_page_config(page_title="Flight Price Prediction", page_icon="🎈", layout="wide")

st.title('🎈 Flight Price Prediction App')

# Cache data loading
@st.cache_data(show_spinner=False)
def load_data():
    df_nf = pd.read_csv("https://raw.githubusercontent.com/arnanbonny/Petproject/master/Clean_Dataset.csv")
    df = df_nf.drop(['flight', 'Unnamed: 0'], axis=1)
    return df

# Cache preprocessing
@st.cache_data(show_spinner=False)
def preprocess_data(_df):
    cat_cols = ['airline','source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city','class']
    
    # Encode categorical variables
    df_encoded = pd.get_dummies(_df, columns=cat_cols, drop_first=True)
    
    # Split features and target
    X = df_encoded.drop('price', axis=1)
    y = df_encoded['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Return column names as well for prediction alignment
    return X_train, X_test, y_train, y_test, X.columns.tolist()

# Cache model training
@st.cache_resource(show_spinner=False)
def train_model(X_train, y_train):
    num_cols = ['duration', 'days_left']
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    
    # Train model with fewer trees for faster loading
    rf_model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train)
    
    return rf_model, scaler

# Cache evaluation
@st.cache_data(show_spinner=False)
def evaluate_model(_model, _scaler, X_test, y_test):
    num_cols = ['duration', 'days_left']
    
    # Scale test data
    X_test_scaled = X_test.copy()
    X_test_scaled[num_cols] = _scaler.transform(X_test[num_cols])
    
    # Make predictions
    y_pred = _model.predict(X_test_scaled)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    return r2, mae, rmse, y_test, y_pred

# Main execution with progress tracking
try:
    with st.spinner('🔄 Loading data...'):
        df = load_data()
    
    with st.spinner('🔄 Preprocessing data...'):
        X_train, X_test, y_train, y_test, feature_columns = preprocess_data(df)
    
    with st.spinner('🔄 Training model...'):
        rf_model, scaler = train_model(X_train, y_train)
    
    with st.spinner('🔄 Evaluating model...'):
        r2, mae, rmse, y_test_eval, y_pred = evaluate_model(rf_model, scaler, X_test, y_test)
    
    # Display results
    st.success('✅ Model trained successfully!')
    
    st.subheader('📊 Model Performance Metrics')
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric('R² Score', f'{r2:.4f}')
    with col2:
        st.metric('MAE', f'₹{mae:.2f}')
    with col3:
        st.metric('RMSE', f'₹{rmse:.2f}')
    
    # Optional: Show data preview
    with st.expander('📋 View Dataset'):
        st.dataframe(df.head(10))
        st.write(f'Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns')
    
    # Optional: Show prediction vs actual plot
    with st.expander('📈 View Prediction Results'):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_test_eval, y_pred, alpha=0.5, s=20)
        ax.plot([y_test_eval.min(), y_test_eval.max()], 
                [y_test_eval.min(), y_test_eval.max()], 'r--', lw=2)
        ax.set_xlabel('Actual Price (₹)', fontsize=12)
        ax.set_ylabel('Predicted Price (₹)', fontsize=12)
        ax.set_title('Actual vs Predicted Flight Prices', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Show sample predictions
        st.subheader('Sample Predictions')
        results_df = pd.DataFrame({
            'Actual Price': y_test_eval.head(10).values,
            'Predicted Price': y_pred[:10],
            'Difference': y_test_eval.head(10).values - y_pred[:10]
        })
        st.dataframe(results_df.style.format({
            'Actual Price': '₹{:.2f}',
            'Predicted Price': '₹{:.2f}',
            'Difference': '₹{:.2f}'
        }))

    # Interactive Prediction Section
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
        training_cols = feature_columns
        
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

except Exception as e:
    st.error(f'❌ An error occurred: {str(e)}')
    st.info('Please check the data source and try again.')
    st.exception(e)
