# Karachi-AQI
This project is about predicting the AQI of Karachi for next 3 days using different ML models.
A real-time Air Quality Index (AQI) monitoring and forecasting system for Karachi using historical data, OpenWeatherMap API, and machine learning models. This project generates synthetic historical features, trains multiple models, and provides live AQI predictions.
The dashboard can be viewed by https://karachiaqibyusman.streamlit.app/. It shows the 3 days prediction using the best trained model.

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
Installation

Clone the repository:

git clone <your-repo-url>
cd <repo-folder>


Create a virtual environment:

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows


Install dependencies:

pip install -r requirements.txt


Create a .env file with your API keys:

OPENWEATHERMAP_API_KEY=your_openweathermap_api_key
HOPSWORKS_API_KEY=your_hopsworks_api_key
HOPSWORKS_PROJECT_NAME=your_hopsworks_project
CITY_NAME=Karachi
CITY_LAT=24.8607
CITY_LON=67.0011
FEATURE_GROUP_NAME=karachi_aqi_features
FEATURE_GROUP_VERSION=1

Usage
1. Generate and store historical features
python generate_synthetic_data.py


Generates 100 days of historical AQI and weather features.

Saves data to Hopsworks Feature Store.

2. Train ML models
python training_pipeline.py


Trains Random Forest, XGBoost, and Neural Network.

Saves trained models to models/ directory.

Selects best model based on RMSE.

3. Run Streamlit dashboard
streamlit run app.py


Provides real-time AQI display.

Shows 3-day ML forecast.

Displays pollutants and health recommendations.

Project Structure
├── app.py                    # Streamlit dashboard
├── generate_synthetic_data.py # Feature generation script
├── training_pipeline.py       # ML model training script
├── models/                    # Saved trained models
├── .env                       # API keys and configuration
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation

Dependencies

Python 3.12+

pandas, numpy

scikit-learn

xgboost

requests

python-dotenv

streamlit

plotly

hopsworks (for feature store)

API & Data Sources

OpenWeatherMap API – real-time AQI & weather data.

Hopsworks Feature Store – historical AQI features storage.

License

This project is licensed under the MIT License.
