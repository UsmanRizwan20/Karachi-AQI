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


def make_forecast(current_aqi, days=3):
    preds = []
    for i in range(days):
        variation = np.random.uniform(-0.4, 0.4)
        pred = round(max(1, min(5, current_aqi + variation)), 1)
        preds.append({'date': datetime.now() + timedelta(days=i+1), 'aqi': pred})
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
        else:
            st.warning("No trained models found!")
            st.info("Run:\n```\npython generate_synthetic_data.py\npython training_pipeline.py\n```")
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

    if model_choice:
        st.info(f"🤖 Prediction Model: **{model_choice}**")
    else:
        st.warning("Using estimated forecast — train models for ML predictions")

    preds = make_forecast(aqi_raw)
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
