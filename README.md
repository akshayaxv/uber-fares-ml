# 🚖 Uber Fare Prediction App

## Overview
The Uber Fare Prediction App is a machine learning-based web application designed to predict the fare for an Uber ride. By inputting the pickup and dropoff locations along with other relevant details, the app provides an estimate of the fare using a pre-trained TensorFlow model.

## Features
* Easy to Use: Simply enter the pickup and dropoff details, along with the date, time, and number of passengers, to get a fare prediction.
* Geocoding: Automatically convert addresses into geographic coordinates using the geopy library.
* Responsive Design: The app layout adjusts based on screen size for optimal user experience.
* Dark Theme: The interface uses a professional dark theme with a modern design.
* Custom Interactions: Features like distance-time interaction and distance-passenger interaction enhance the prediction accuracy.

## Tech Stack
Frontend: Streamlit for the user interface
Backend: TensorFlow for fare prediction
Geocoding: geopy for converting addresses to geographic coordinates
Language: Python
Setup and Installation
Prerequisites
Ensure you have the following installed:
 1) Python 3.7 or later
 2) Pip (Python package manager)

## Installation
1) Clone the repository:
   git clone https://github.com/Hemasri05/uber-fares-ml.git
   cd uber-fares-ml

2) Install the required packages:
   pip install -r requirements.txt

3) Place the model file:
   Ensure the model.h5 file (your trained model) is in the root directory of the project.

4) Run the app:
  streamlit run app.py

## How It Works :-
1) User Inputs: The user provides details such as pickup date, time, pickup and dropoff addresses, and the number of passengers.
2) Geocoding: The app converts the provided addresses into longitude and latitude using the geopy library.
3) Prediction: The app calculates additional features like the distance traveled and time of day, and then feeds the data into the TensorFlow model to predict the fare.
4) Output: The predicted fare is displayed on the screen.

## Contact
For any inquiries, please contact myhema05@gmail.com , akshaya.07.sep@gmail.com


