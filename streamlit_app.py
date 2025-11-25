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


with st.expander('Data'):
  st.write('Raw Dataset')
  df = pd.read_csv('https://raw.githubusercontent.com/arnanbonny/Petproject/master/Clean_Dataset.csv')
  df
