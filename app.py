import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# Set page config to wide mode and dark theme with an emoji
st.set_page_config(layout="wide", page_title="Uber Fare Prediction App", page_icon="🚖")

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model.h5', custom_objects={'mse': tf.keras.losses.MeanSquaredError()})

model = load_model()

# Streamlit app
def main():
    # Custom CSS for dark theme and layout
    st.markdown("""
    <style>
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        .stTextInput > div > div > input, .stNumberInput > div > div > input, .stDateInput > div > div > input, .stTimeInput > div > div > input {
            background-color: #262730;
            color: #FAFAFA;
        }
        .stButton > button {
            background-color: #FF4B4B;
            color: white;
        }
        .stTitle {
            color: #FFA500;
        }
        .stMenuItem {
            background-color: #262730;
            color: #FAFAFA;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("Uber Fare Prediction App 🚖")

    st.write("Enter the ride details to predict the fare:")

    # Creating columns for layout with adjusted ratios to fill the page
    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
    col5, col6, col7, col8 = st.columns([2, 2, 2, 2])
    
    with col1:
        pickup_date = st.date_input("Pickup Date")
    with col2:
        pickup_time = st.time_input("Pickup Time")
    
    with col3:
        street = st.text_input("Pickup Street", "75 Bay Street")
    with col4:
        city = st.text_input("Pickup City", "Toronto")
    
    with col5:
        province = st.text_input("Pickup Province", "Ontario")
    with col6:
        country = st.text_input("Pickup Country", "Canada")
    
    with col7:
        dropoff_street = st.text_input("Dropoff Street", "100 Queen Street")
    with col8:
        dropoff_city = st.text_input("Dropoff City", "Toronto")
    
    col9, col10 = st.columns([2, 2])
    
    with col9:
        dropoff_province = st.text_input("Dropoff Province", "Ontario")
    with col10:
        dropoff_country = st.text_input("Dropoff Country", "Canada")
    
    col11, col12 = st.columns([2, 2])
    
    with col11:
        passenger_count = st.number_input("Number of Passengers", min_value=1, max_value=6, value=1, step=1)

    # Geocoding for pickup location
    pickup_longitude, pickup_latitude = get_coordinates(street, city, province, country)

    # Geocoding for dropoff location
    dropoff_longitude, dropoff_latitude = get_coordinates(dropoff_street, dropoff_city, dropoff_province, dropoff_country)

    if st.button("Predict Fare"):
        # Combine pickup date and time
        pickup_datetime = datetime.combine(pickup_date, pickup_time)
        
        # Prepare input data
        input_data = prepare_input(pickup_datetime, pickup_longitude, pickup_latitude,
                                   dropoff_longitude, dropoff_latitude, passenger_count)
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Display result
        st.success(f"The predicted fare is: ${prediction[0][0]:.2f}")

def prepare_input(pickup_datetime, pickup_longitude, pickup_latitude,
                  dropoff_longitude, dropoff_latitude, passenger_count):
    # Create a DataFrame with the input data
    input_df = pd.DataFrame({
        'pickup_longitude': [pickup_longitude],
        'pickup_latitude': [pickup_latitude],
        'dropoff_longitude': [dropoff_longitude],
        'dropoff_latitude': [dropoff_latitude],
        'passenger_count': [passenger_count],
        'hour': [pickup_datetime.hour],
        'day': [pickup_datetime.day],
        'month': [pickup_datetime.month],
        'year': [pickup_datetime.year],
        'dayofweek': [pickup_datetime.weekday()]
    })
    
    # Calculate distance
    input_df['dist_travel_km'] = input_df.apply(
    lambda row: haversine_distance(
        row['pickup_latitude'],
        row['pickup_longitude'],
        row['dropoff_latitude'],
        row['dropoff_longitude']), axis=1)

    
    # Calculate derived features
    input_df['distance_time_interaction'] = input_df['dist_travel_km'] * input_df['hour']
    input_df['distance_passenger_interaction'] = input_df['dist_travel_km'] * input_df['passenger_count']
    input_df['log_dist_travel_km'] = np.log1p(input_df['dist_travel_km'])
    
    # Ensure all features are present and in the correct order
    features = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 
                'passenger_count', 'hour', 'day', 'month', 'year', 'dayofweek', 'dist_travel_km',
                'distance_time_interaction', 'distance_passenger_interaction', 'log_dist_travel_km']
    
    input_array = input_df[features].to_numpy()
    
    return input_array

def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])  # Ensure values are float
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance_km = 6371 * c
    return distance_km


def get_coordinates(street, city, province, country):
    geolocator = Nominatim(user_agent="uber-fare-prediction")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, max_retries=3, error_wait_seconds=10)
    try:
        location = geolocator.geocode(f"{street}, {city}, {province}, {country}", timeout=10)
        if location:
            return location.longitude, location.latitude
        else:
            st.error("Location not found. Please check the address.")
            return None, None
    except GeocoderUnavailable: # type: ignore
        st.error("Geocoding service is currently unavailable. Please try again later.")
        return None, None


if __name__ == "__main__":
    main()
