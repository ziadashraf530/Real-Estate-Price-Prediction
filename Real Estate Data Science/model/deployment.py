import streamlit as st
import pickle
import json
import numpy as np

# Load the trained model
with open('banglore_home_prices_model.pickle', 'rb') as f:
    model = pickle.load(f)

# Load column information
with open('columns.json', 'r') as f:
    data_columns = json.load(f)['data_columns']

# Extract location names (skip first 3 columns: total_sqft, bath, bhk)
locations = data_columns[3:]

def predict_price(location, sqft, bath, bhk):
    """Predict house price based on user inputs"""
    try:
        loc_index = data_columns.index(location.lower())
    except:
        loc_index = -1
    
    # Create input array with zeros
    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    
    # Set location column to 1 (one-hot encoding)
    if loc_index >= 0:
        x[loc_index] = 1
    
    return round(model.predict([x])[0], 2)

# Streamlit UI
st.title('🏠 Bangalore Real Estate Price Predictor')
st.write('Enter property details to get an estimated price')

# Create input fields
col1, col2 = st.columns(2)

with col1:
    location = st.selectbox('Select Location', sorted(locations))
    sqft = st.number_input('Total Square Feet', min_value=300, max_value=30000, value=1000, step=50)

with col2:
    bhk = st.number_input('Number of Bedrooms (BHK)', min_value=1, max_value=16, value=2, step=1)
    bath = st.number_input('Number of Bathrooms', min_value=1, max_value=16, value=2, step=1)

# Predict button
if st.button('Predict Price', type='primary'):
    price = predict_price(location, sqft, bath, bhk)
    st.success(f'### Estimated Price: ₹ {price} Lakhs')
    st.info(f'📍 Location: {location.title()}\n\n📏 Area: {sqft} sqft\n\n🛏️ {bhk} BHK, {bath} Bath')
