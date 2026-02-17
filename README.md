# Karachi-AQI
This project is about predicting the AQI of Karachi for next 3 days using different ML models.
A real-time Air Quality Index (AQI) monitoring and forecasting system for Karachi using historical data, OpenWeatherMap API, and machine learning models. This project generates synthetic historical features, trains multiple models, and provides live AQI predictions.

Features

Fetches real-time AQI and weather data from OpenWeatherMap API.

Generates synthetic historical features for model training.

Trains three ML models:

Random Forest Regressor

XGBoost Regressor

Neural Network (MLPRegressor)

Predicts 3-day AQI forecast.

Supports Hopsworks Feature Store for storing historical features.

Provides pollutant breakdown and weather context.

Streamlit dashboard for live AQI visualization.
