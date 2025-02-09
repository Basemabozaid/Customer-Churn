# -*- coding: utf-8 -*-
"""Untitled18.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1GOk-6TocUyr7fYUoHdrVIsHQbwd4s43I
"""

import streamlit as st
import pickle
import pandas as pd

with open('log_reg.pkl', 'rb') as file:
    model = pickle.load(file)

with open('transformer.pkl', 'rb') as file:
    trans = pickle.load(file)

# Streamlit app
st.title('Customer Churn Prediction')

# Input form for customer details
contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
onlinesecurity = st.selectbox('Online Security', ['Yes', 'No'])
techsupport = st.selectbox('Tech Support', ['Yes', 'No'])
internetservice = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
onlinebackup = st.selectbox('Online Backup', ['Yes', 'No'])
tenure = st.number_input('Tenure (Months)', min_value=0, max_value=100, value=1)
monthlycharges = st.number_input('Monthly Charges', min_value=0.0, value=50.0)
totalcharges = st.number_input('Total Charges', min_value=0.0, value=100.0)

# Create the customer data dictionary
cust = {
    'contract': contract,
    'onlinesecurity': onlinesecurity,
    'techsupport': techsupport,
    'internetservice': internetservice,
    'onlinebackup': onlinebackup,
    'tenure': tenure,
    'monthlycharges': monthlycharges,
    'totalcharges': totalcharges
}

# Predict churn
if st.button('Predict'):
    # Convert input into DataFrame and ensure all columns are present
    cust_df = pd.DataFrame([cust])

    if 'seniorcitizen' not in cust_df:
        cust_df['seniorcitizen'] = 0  # Adding missing column

    # Apply transformation
    cust_transformed = trans.transform(cust_df)

    # Predict churn
    prediction = model.predict(cust_transformed)[0]

    # Display result
    if prediction == 0:
        st.success('The customer is NOT likely to churn.')
    else:
        st.error('The customer is likely to churn.')

