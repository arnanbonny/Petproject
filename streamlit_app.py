import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


st.title('🎈 App Name')


df_nf = pd.read_csv("https://raw.githubusercontent.com/arnanbonny/Petproject/master/Clean_Dataset.csv")
df = df_nf.drop(['flight', 'Unnamed: 0'], axis=1)
cat_cols = ['airline','source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city','class']
num_cols = ['duration', 'days_left']
target= 'price'
df_encoded = pd.get_dummies(df, columns = cat_cols, drop_first=True)
X = df_encoded.drop('price', axis = 1)
y = df_encoded['price']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
scaler = StandardScaler()
X_train[num_cols]= scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

rf_model = RandomForestRegressor(n_estimators = 100, random_state =42)
rf_model.fit(X_train, y_train)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
y_pred = rf_model.predict(X_test)
mae =  mean_absolute_error(y_test, y_pred)
rmse =  mean_squared_error(y_test, y_pred, squared = False)
r2 = r2_score(y_test, y_pred)


st.write(r2)
