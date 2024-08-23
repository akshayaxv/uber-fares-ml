import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime, time

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model.h5', custom_objects={'mse': tf.keras.losses.MeanSquaredError()})

model = load_model()

# Streamlit app
def main():
    st.title("Uber Fare Prediction App")

    st.write("Enter the ride details to predict the fare:")

    # Input fields
    pickup_date = st.date_input("Pickup Date")
    pickup_time = st.time_input("Pickup Time", value=time(12, 0))
    pickup_datetime = datetime.combine(pickup_date, pickup_time)
    
    pickup_longitude = st.number_input("Pickup Longitude", value=-73.98)
    pickup_latitude = st.number_input("Pickup Latitude", value=40.75)
    dropoff_longitude = st.number_input("Dropoff Longitude", value=-73.98)
    dropoff_latitude = st.number_input("Dropoff Latitude", value=40.75)
    passenger_count = st.number_input("Number of Passengers", min_value=1, max_value=6, value=1, step=1)

    if st.button("Predict Fare"):
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
    input_df['dist_travel_km'] = haversine_distance(
        input_df['pickup_latitude'], input_df['pickup_longitude'],
        input_df['dropoff_latitude'], input_df['dropoff_longitude']
    )
    
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
    R = 6371  # Earth's radius in kilometers

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    distance = R * c
    return distance

if __name__ == "__main__":
    main()