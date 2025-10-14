import streamlit as st

import pickle

import json

import numpy as np

import os



# -------------------------------

# 1️⃣ Load model safely

# -------------------------------

@st.cache_resource

def load_model():

    try:

        with open('banglore_home_prices_model.pickle', 'rb') as f:

            model = pickle.load(f)

        return model

    except FileNotFoundError:

        st.error("❌ Model file not found! Please make sure 'banglore_home_prices_model.pickle' exists.")

        st.stop()



# -------------------------------

# 2️⃣ Load columns safely

# -------------------------------

@st.cache_data

def load_columns():

    try:

        with open('columns.json', 'r') as f:

            data_columns = json.load(f)['data_columns']

        return data_columns

    except FileNotFoundError:

        st.error("❌ Columns file not found! Please make sure 'columns.json' exists.")

        st.stop()

    except KeyError:

        st.error("⚠️ 'data_columns' key missing in columns.json.")

        st.stop()



# Load data

model = load_model()

data_columns = load_columns()

locations = data_columns[3:]  # skip sqft, bath, bhk



# -------------------------------

# 3️⃣ Prediction function

# -------------------------------

def predict_price(location, sqft, bath, bhk):

    """Predict house price based on user inputs."""

    try:

        loc_index = data_columns.index(location.lower())

    except ValueError:

        loc_index = -1

    

    # Create input array

    x = np.zeros(len(data_columns))

    x[0] = sqft

    x[1] = bath

    x[2] = bhk



    if loc_index >= 0:

        x[loc_index] = 1



    try:

        prediction = model.predict([x])[0]

        return round(prediction, 2)

    except Exception as e:

        st.error(f"Prediction error: {e}")

        return None



# -------------------------------

# 4️⃣ Streamlit UI

# -------------------------------

st.set_page_config(page_title="🏠 Bangalore Price Predictor", layout="centered")



st.title("🏡 Bangalore Real Estate Price Predictor")

st.write("Enter the details below to estimate the property price in Bangalore.")



# Use columns for better layout

col1, col2 = st.columns(2)



with col1:

    location = st.selectbox('📍 Select Location', sorted(locations))

    sqft = st.number_input('📏 Total Square Feet', min_value=300, max_value=30000, value=1000, step=50)



with col2:

    bhk = st.number_input('🛏 Number of Bedrooms (BHK)', min_value=1, max_value=16, value=2, step=1)

    bath = st.number_input('🚿 Number of Bathrooms', min_value=1, max_value=16, value=2, step=1)



# Predict button

if st.button('🔮 Predict Price', type='primary', use_container_width=True):

    with st.spinner("Calculating..."):

        price = predict_price(location, sqft, bath, bhk)

    

    if price is not None:

        st.success(f"### 💰 Estimated Price: ₹ {price} Lakhs")

        st.info(

            f"**Details:**\n"

            f"- Location: {location.title()}\n"

            f"- Area: {sqft} sqft\n"

            f"- {bhk} BHK, {bath} Bath"

        )



# -------------------------------

# 5️⃣ Footer

# -------------------------------

st.markdown("---")

st.caption("Developed by **Magid Yaseer** | Machine Learning & AI Specialist ⚡")
