# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 17:20:25 2024

@author: HQDS
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import joblib
from sklearn.linear_model import LinearRegression
import requests
import pickle



# Title of the app
st.title("AI Model Testing App")

# Sidebar for input parameters
st.sidebar.header("Input Parameters")

# Function to get user input
def user_input_features():
    A_Age = st.sidebar.slider('A_Age', 0, 300, 70)
    B_MatchingTcpa = st.sidebar.slider('B_MatchingTcpa', 0.0, 2.0, 0.5)
    C_MatchingAdcopy = st.sidebar.slider('C_MatchingAdcopy', 0.0, 2.0, 0.5)
    D_Distance_Factor = st.sidebar.slider('D_Distance_Factor', 0.0, 10000.0, 1000.0)
    E_Rate_of_lead_ingestion = st.sidebar.slider('E_Rate_of_lead_ingestion', 0.0, 50.0, 1.0)
    data = {'A_Age': A_Age,
            'B_MatchingTcpa': B_MatchingTcpa,
            'C_MatchingAdcopy': C_MatchingAdcopy,
            'D_Distance_Factor': D_Distance_Factor,
            'E_Rate_of_lead_ingestion': E_Rate_of_lead_ingestion}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Display the input parameters
st.subheader('Input Parameters')
st.write(df)


# Step 2: Load the .pkl file into a Python variable using joblib
model = joblib.load('random_forest_regressor_model.pkl')


y_pred = model.predict(df)

st.subheader('Prediction')
st.write(y_pred)
