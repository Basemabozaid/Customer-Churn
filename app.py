import streamlit as st
import pickle
import pandas as pd
import urllib.request
import os

# Define GitHub raw URLs for model and transformer
GITHUB_BASE_URL = "https://github.com/Basemabozaid/Customer-Churn/tree/main/models"

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

st.title('Customer Churn Prediction')
