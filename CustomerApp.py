import streamlit as st
import pickle
import pandas as pd
import urllib.request
import os

# Define GitHub raw URLs for model and transformer
GITHUB_BASE_URL = "https://raw.githubusercontent.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY_NAME/main/models/"

# File names
transformer_filename = "transformer.pkl"
model_filename = "log_reg.pkl"

# Function to download files from GitHub
def download_file(filename):
    url = GITHUB_BASE_URL + filename
    local_path = filename  # Saves the file in the same directory as app.py
    if not os.path.exists(local_path):  # Download only if not already present
        urllib.request.urlretrieve(url, local_path)
    return local_path

# Download and load transformer
trans_path = download_file(transformer_filename)
with open(trans_path, "rb") as f:
    trans = pickle.load(f)

# Download and load model
model_path = download_file(model_filename)
with open(model_path, "rb") as f:
    model = pickle.load(f)
    
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

