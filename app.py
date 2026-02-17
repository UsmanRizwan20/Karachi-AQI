import os
from pathlib import Path
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import joblib
from dotenv import load_dotenv

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")
KARACHI_LAT  = float(os.getenv("CITY_LAT", "24.8607"))
KARACHI_LON  = float(os.getenv("CITY_LON", "67.0011"))
CITY_NAME    = os.getenv("CITY_NAME", "Karachi")
MODEL_DIR    = Path("models")

st.set_page_config(page_title=f"{CITY_NAME} AQI Predictor", page_icon="🌍", layout="wide")

st.markdown("""
<style>
    h1 { font-weight: 700; }
    .stMetric { background-color: #1E2433; padding: 1.2rem; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


def get_aqi_category(aqi):
    if aqi <= 50:    return "Good", "#00E400", "😊"
    elif aqi <= 100: return "Moderate", "#FFFF00", "😐"
    elif aqi <= 150: return "Unhealthy for Sensitive Groups", "#FF7E00", "😷"
    elif aqi <= 200: return "Unhealthy", "#FF0000", "☹️"
    elif aqi <= 300: return "Very Unhealthy", "#8F3F97", "😨"
    else:            return "Hazardous", "#7E0023", "☠️"


@st.cache_data(ttl=3600)
def fetch_current_aqi():
    try:
        aqi_r = requests.get("http://api.openweathermap.org/data/2.5/air_pollution",
            params={'lat': KARACHI_LAT, 'lon': KARACHI_LON, 'appid': OPENWEATHER_API_KEY}, timeout=10)
        aqi_r.raise_for_status()

        wx_r = requests.get("http://api.openweathermap.org/data/2.5/weather",
            params={'lat': KARACHI_LAT, 'lon': KARACHI_LON,
                    'appid': OPENWEATHER_API_KEY, 'units': 'metric'}, timeout=10)
        wx_r.raise_for_status()

        c = aqi_r.json()['list'][0]['components']
        w = wx_r.json()

        return {
            'aqi':         aqi_r.json()['list'][0]['main']['aqi'],
            'pm2_5':       c.get('pm2_5', 0),
            'pm10':        c.get('pm10', 0),
            'no2':         c.get('no2', 0),
            'o3':          c.get('o3', 0),
            'so2':         c.get('so2', 0),
            'co':          c.get('co', 0),
            'temperature': w['main']['temp'],
            'humidity':    w['main']['humidity'],
            'wind_speed':  w['wind']['speed'],
            'pressure':    w['main']['pressure'],
        }
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


@st.cache_resource
def load_models():
    models = {}
    for name, path in [
        ('Random Forest',    MODEL_DIR / "random_forest_model.pkl"),
        ('XGBoost',          MODEL_DIR / "xgboost_model.pkl"),
    ]:
        if path.exists():
            models[name] = joblib.load(path)

    nn_path     = MODEL_DIR / "neural_network_model.pkl"
    scaler_path = MODEL_DIR / "neural_network_scaler.pkl"
    if nn_path.exists() and scaler_path.exists():
        models['Neural Network (MLP)'] = {
            'model':  joblib.load(nn_path),
            'scaler': joblib.load(scaler_path)
        }
    return models


def build_feature_row(data, day_offset=0):
    """
    Build a feature row from current API data.
    Includes all features the model was trained on.
    Day-over-day values are realistically shifted for multi-day forecasts.
    """
    future_date = datetime.now() + timedelta(days=day_offset)

    hour        = future_date.hour
    day         = future_date.day
    month       = future_date.month
    day_of_week = future_date.weekday()       # 0=Monday, 6=Sunday
    is_weekend  = int(day_of_week >= 5)
    is_rush_hour = int(hour in range(7, 10) or hour in range(17, 20))

    # Season: 0=winter, 1=spring, 2=summer, 3=fall
    season = (month % 12) // 3

    # Deterministic day-over-day trends (no randomness)
    temp_trend     = day_offset * 0.3
    humidity_trend = day_offset * 0.5

    temperature = data['temperature'] + temp_trend
    feels_like  = temperature - 2.0          # approximate feels_like
    temp_min    = temperature - 3.0
    temp_max    = temperature + 3.0
    temp_range  = temp_max - temp_min
    humidity    = min(100, data['humidity'] + humidity_trend)

    # Cyclical hour encoding
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    pm2_5 = data['pm2_5']
    pm10  = data['pm10']
    pm_ratio = pm2_5 / pm10 if pm10 > 0 else 0.0

    row = {
        'hour':        hour,
        'day':         day,
        'month':       month,
        'day_of_week': day_of_week,
        'is_weekend':  is_weekend,
        'season':      season,
        'pm2_5':       pm2_5,
        'pm10':        pm10,
        'no2':         data['no2'],
        'o3':          data['o3'],
        'so2':         data['so2'],
        'co':          data['co'],
        'temperature': temperature,
        'feels_like':  feels_like,
        'temp_min':    temp_min,
        'temp_max':    temp_max,
        'pressure':    data['pressure'],
        'humidity':    humidity,
        'wind_speed':  data['wind_speed'],
        'wind_deg':    0.0,   # not available from API, use neutral default
        'clouds':      0.0,   # not available from API, use neutral default
        'pm_ratio':    pm_ratio,
        'temp_range':  temp_range,
        'is_rush_hour': is_rush_hour,
        'hour_sin':    hour_sin,
        'hour_cos':    hour_cos,
    }
    return pd.DataFrame([row])


def predict_with_model(model_obj, feature_df):
    """Run prediction handling both plain models and NN (which needs a scaler)."""
    if isinstance(model_obj, dict):
        # Neural Network with scaler
        scaled = model_obj['scaler'].transform(feature_df)
        return float(model_obj['model'].predict(scaled)[0])
    else:
        return float(model_obj.predict(feature_df)[0])


def make_forecast(data, model_obj, days=3):
    """Generate a 3-day forecast using the actual trained ML model."""
    preds = []
    for i in range(1, days + 1):
        feature_df = build_feature_row(data, day_offset=i)
        try:
            predicted_aqi = predict_with_model(model_obj, feature_df)
            predicted_aqi = round(max(1.0, min(5.0, predicted_aqi)), 2)
        except Exception as e:
            st.warning(f"Prediction error on day {i}: {e}")
            predicted_aqi = data['aqi']
        preds.append({
            'date': datetime.now() + timedelta(days=i),
            'aqi':  predicted_aqi
        })
    return preds


def main():
    st.title(f"🌍 {CITY_NAME} Air Quality Index Predictor")
    st.markdown("### Real-time AQI Monitoring & 3-Day ML Forecast")
    st.markdown("---")

    with st.sidebar:
        st.header("⚙️ Settings")
        st.markdown("### 🤖 ML Models")
        models = load_models()

        if models:
            best_model_file = MODEL_DIR / "best_model.txt"

            if best_model_file.exists():
                with open(best_model_file, "r") as f:
                    best_model_name = f.read().strip()
                if best_model_name in models:
                    model_choice = best_model_name
                    st.success(f"🏆 Best Model Auto-Selected: {model_choice}")
                else:
                    model_choice = list(models.keys())[0]
                    st.warning("Best model not found — using first available.")
            else:
                model_choice = list(models.keys())[0]
                st.warning("best_model.txt not found — using first available model.")

            # Let user manually override model choice
            model_choice = st.selectbox("Choose Model", list(models.keys()),
                                        index=list(models.keys()).index(model_choice))
        else:
            st.warning("No trained models found!")
            st.info("Run:\n```\npython training_pipeline.py\n```")
            model_choice = None

        st.markdown("---")
        st.markdown("### 📍 Location")
        st.write(f"**City:** {CITY_NAME}")
        st.write(f"**Lat:** {KARACHI_LAT} | **Lon:** {KARACHI_LON}")

        st.markdown("---")
        st.markdown("### 🔬 Models")
        st.write("1️⃣ Random Forest")
        st.write("2️⃣ XGBoost")
        st.write("3️⃣ Neural Network (MLP)")

        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()

    with st.spinner("Fetching live AQI data for Karachi..."):
        data = fetch_current_aqi()

    if not data:
        st.error("Cannot fetch data. Check OPENWEATHERMAP_API_KEY in .env file.")
        return

    st.header("📊 Current Air Quality")

    aqi_raw     = data['aqi']
    aqi_display = aqi_raw * 50
    cat, color, emoji = get_aqi_category(aqi_display)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AQI Index",    aqi_raw)
    c2.metric("PM2.5",        f"{data['pm2_5']:.1f} μg/m³")
    c3.metric("Temperature",  f"{data['temperature']:.1f}°C")
    c4.metric("Humidity",     f"{data['humidity']:.0f}%")

    st.markdown(f"""
    <div style='padding:1rem; background:{color}; color:black; border-radius:8px; margin:1rem 0;'>
        <h3>{emoji} Air Quality: {cat}</h3>
        <p>Current AQI for Karachi is {aqi_raw} — {cat}</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    st.header("📅 3-Day AQI Forecast")

    if model_choice and models:
        st.info(f"🤖 Predictions generated by: **{model_choice}**")
        preds = make_forecast(data, models[model_choice])
    else:
        st.warning("No model available for forecast.")
        return

    cols  = st.columns(3)
    for i, pred in enumerate(preds):
        with cols[i]:
            st.subheader(pred['date'].strftime("%A"))
            st.metric(pred['date'].strftime("%b %d"), f"AQI {pred['aqi']}")
            cat2, _, em2 = get_aqi_category(pred['aqi'] * 50)
            st.markdown(f"{em2} **{cat2}**")

    st.markdown("---")
    st.subheader("📈 AQI Trend")

    dates  = ['Today'] + [p['date'].strftime("%a") for p in preds]
    values = [aqi_raw]  + [p['aqi'] for p in preds]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=values,
        mode='lines+markers',
        line=dict(color='#00E400', width=3),
        marker=dict(size=12)
    ))
    fig.update_layout(
        title='AQI Forecast - Karachi',
        xaxis_title='Day', yaxis_title='AQI',
        height=400,
        plot_bgcolor='#1A1F2E', paper_bgcolor='#1A1F2E',
        font=dict(color='white')
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.header("🔬 Pollutant Breakdown")

    fig2 = go.Figure(data=[go.Bar(
        x=['PM2.5', 'PM10', 'NO₂', 'O₃', 'SO₂', 'CO'],
        y=[data['pm2_5'], data['pm10'], data['no2'],
           data['o3'],    data['so2'], data['co']],
        marker_color=['#e74c3c','#e67e22','#f1c40f','#3498db','#9b59b6','#1abc9c']
    )])
    fig2.update_layout(
        title='Current Pollutant Levels (μg/m³)',
        height=400,
        plot_bgcolor='#1A1F2E', paper_bgcolor='#1A1F2E',
        font=dict(color='white')
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.header("💡 Health Recommendations")

    if   aqi_display <= 50:  st.success("✅ Air quality is GOOD. Enjoy outdoor activities!")
    elif aqi_display <= 100: st.info("ℹ️ MODERATE. Sensitive people should be cautious.")
    elif aqi_display <= 150: st.warning("⚠️ UNHEALTHY for sensitive groups. Reduce outdoor activity.")
    elif aqi_display <= 200: st.error("🚨 UNHEALTHY. Everyone should limit outdoor exertion.")
    else:                    st.error("☠️ HAZARDOUS! Avoid all outdoor activities!")

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; color:#888; padding:2rem;'>
        <p>🤖 Models: Random Forest | XGBoost | Neural Network (MLP)</p>
        <p>📡 Live data: OpenWeatherMap API | 🗄️ Feature Store: Hopsworks</p>
        <p>🌍 Karachi AQI Predictor — Serverless ML Pipeline</p>
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    if not OPENWEATHER_API_KEY:
        st.error("❌ OPENWEATHERMAP_API_KEY not found in .env file!")
    else:
        main()
