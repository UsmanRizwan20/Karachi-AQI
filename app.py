import os
import json
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


def pm25_to_us_aqi(pm25: float) -> float:
    """Convert PM2.5 concentration (μg/m³) to US AQI using EPA breakpoints."""
    breakpoints = [
        (0.0,   12.0,    0,   50),
        (12.1,  35.4,   51,  100),
        (35.5,  55.4,  101,  150),
        (55.5, 150.4,  151,  200),
        (150.5, 250.4, 201,  300),
        (250.5, 350.4, 301,  400),
        (350.5, 500.4, 401,  500),
    ]
    pm25 = max(0.0, min(500.4, pm25))
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if c_lo <= pm25 <= c_hi:
            return round((i_hi - i_lo) / (c_hi - c_lo) * (pm25 - c_lo) + i_lo, 1)
    return 500.0


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

        # Fetch 5-day forecast to get realistic daily weather per day
        fc_r = requests.get("http://api.openweathermap.org/data/2.5/forecast",
            params={'lat': KARACHI_LAT, 'lon': KARACHI_LON,
                    'appid': OPENWEATHER_API_KEY, 'units': 'metric', 'cnt': 40}, timeout=10)
        fc_r.raise_for_status()

        c = aqi_r.json()['list'][0]['components']
        w = wx_r.json()
        forecast_list = fc_r.json().get('list', [])

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
            'wind_deg':    w['wind'].get('deg', 0),
            'pressure':    w['main']['pressure'],
            'clouds':      w['clouds']['all'],
            'forecast':    forecast_list,
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


@st.cache_data
def load_feature_names():
    fn_path = MODEL_DIR / "feature_names.json"
    if fn_path.exists():
        with open(fn_path) as f:
            return json.load(f)
    return None


def get_forecast_weather_for_day(forecast_list, day_offset):
    """Extract representative noon weather for a given day offset from OWM forecast."""
    target_date = (datetime.now() + timedelta(days=day_offset)).date()
    day_entries = [
        e for e in forecast_list
        if datetime.fromtimestamp(e['dt']).date() == target_date
    ]
    if not day_entries:
        return None
    noon_entries = [e for e in day_entries if datetime.fromtimestamp(e['dt']).hour == 12]
    entry = noon_entries[0] if noon_entries else day_entries[len(day_entries) // 2]
    temp_vals = [e['main']['temp'] for e in day_entries]
    return {
        'temperature': entry['main']['temp'],
        'feels_like':  entry['main']['feels_like'],
        'temp_min':    min(temp_vals),
        'temp_max':    max(temp_vals),
        'pressure':    entry['main']['pressure'],
        'humidity':    entry['main']['humidity'],
        'wind_speed':  entry['wind']['speed'],
        'wind_deg':    entry['wind'].get('deg', 0),
        'clouds':      entry['clouds']['all'],
    }


def build_feature_row(data, day_offset, prev_pm25=None):
    """
    Build one feature row for the model at `day_offset` days ahead.
    Each day gets DIFFERENT pollutant and weather values so predictions vary.
    """
    future_date  = datetime.now() + timedelta(days=day_offset)
    feature_names = load_feature_names()

    hour         = 12   # predict for midday
    day          = future_date.day
    month        = future_date.month
    day_of_week  = future_date.weekday()
    is_weekend   = int(day_of_week >= 5)
    is_rush_hour = 0
    season       = (month % 12) // 3
    hour_sin     = np.sin(2 * np.pi * hour / 24)
    hour_cos     = np.cos(2 * np.pi * hour / 24)

    # ── PM2.5: mean-reversion to seasonal baseline ────────────────────────────
    SEASONAL_BASELINE_PM25 = {
        1: 55, 2: 50, 3: 45, 4: 40, 5: 38, 6: 35,
        7: 33, 8: 33, 9: 38, 10: 45, 11: 52, 12: 58
    }
    pm25_baseline   = SEASONAL_BASELINE_PM25.get(month, 45)
    base_pm25       = prev_pm25 if prev_pm25 is not None else data['pm2_5']
    pm2_5           = base_pm25 + 0.15 * (pm25_baseline - base_pm25)

    pm10_ratio = (data['pm10'] / data['pm2_5']) if data['pm2_5'] > 0 else 1.65
    pm10_ratio = max(1.0, min(3.0, pm10_ratio))
    pm10 = pm2_5 * pm10_ratio

    decay = 0.95 ** day_offset
    no2 = data['no2'] * decay
    o3  = data['o3']  * (0.97 ** day_offset)
    so2 = data['so2'] * decay
    co  = data['co']  * decay

    pm_ratio        = pm2_5 / pm10 if pm10 > 0 else 0.0
    aqi_change_rate = pm2_5 - base_pm25

    # ── Weather: use OWM 5-day forecast ───────────────────────────────────────
    wx = get_forecast_weather_for_day(data.get('forecast', []), day_offset)
    if wx:
        temperature = wx['temperature']
        feels_like  = wx['feels_like']
        temp_min    = wx['temp_min']
        temp_max    = wx['temp_max']
        pressure    = wx['pressure']
        humidity    = wx['humidity']
        wind_speed  = wx['wind_speed']
        wind_deg    = wx['wind_deg']
        clouds      = wx['clouds']
    else:
        temperature = data['temperature'] + day_offset * 0.3
        feels_like  = temperature - 2.0
        temp_min    = temperature - 3.0
        temp_max    = temperature + 3.0
        pressure    = data['pressure']
        humidity    = min(100, data['humidity'] + day_offset * 0.5)
        wind_speed  = data['wind_speed']
        wind_deg    = data.get('wind_deg', 0)
        clouds      = data.get('clouds', 0)

    temp_range = temp_max - temp_min

    row = {
        'hour':             hour,
        'day':              day,
        'month':            month,
        'day_of_week':      day_of_week,
        'is_weekend':       is_weekend,
        'season':           season,
        'pm2_5':            pm2_5,
        'pm10':             pm10,
        'no2':              no2,
        'o3':               o3,
        'so2':              so2,
        'co':               co,
        'temperature':      temperature,
        'feels_like':       feels_like,
        'temp_min':         temp_min,
        'temp_max':         temp_max,
        'pressure':         pressure,
        'humidity':         humidity,
        'wind_speed':       wind_speed,
        'wind_deg':         wind_deg,
        'clouds':           clouds,
        'pm_ratio':         pm_ratio,
        'temp_range':       temp_range,
        'is_rush_hour':     is_rush_hour,
        'hour_sin':         hour_sin,
        'hour_cos':         hour_cos,
        'aqi_change_rate':  aqi_change_rate,
    }

    df = pd.DataFrame([row])

    # Align columns exactly to what the model was trained on
    if feature_names:
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0.0
        df = df[feature_names]

    return df, pm2_5


def predict_with_model(model_obj, feature_df):
    if isinstance(model_obj, dict):
        scaled = model_obj['scaler'].transform(feature_df)
        return float(model_obj['model'].predict(scaled)[0])
    else:
        return float(model_obj.predict(feature_df)[0])


def make_forecast(data, model_obj, days=3):
    """
    Autoregressive 3-day forecast: model predicts PM2.5, converted to US AQI.
    Each day's predicted PM2.5 feeds into the next day's feature row.
    """
    preds = []
    prev_pm25 = data['pm2_5']

    for i in range(1, days + 1):
        feature_df, projected_pm25 = build_feature_row(data, day_offset=i, prev_pm25=prev_pm25)
        try:
            predicted_pm25 = predict_with_model(model_obj, feature_df)
            predicted_pm25 = max(0.0, predicted_pm25)
            predicted_aqi  = pm25_to_us_aqi(predicted_pm25)
        except Exception as e:
            st.warning(f"Prediction error on day {i}: {e}")
            predicted_pm25 = projected_pm25
            predicted_aqi  = pm25_to_us_aqi(projected_pm25)

        prev_pm25 = predicted_pm25   # chain predictions

        preds.append({
            'date':  datetime.now() + timedelta(days=i),
            'aqi':   predicted_aqi,
            'pm2_5': round(predicted_pm25, 1),
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
                model_choice = best_model_name if best_model_name in models else list(models.keys())[0]
                if best_model_name in models:
                    st.success(f"🏆 Best Model Auto-Selected: {model_choice}")
                else:
                    st.warning("Best model not found — using first available.")
            else:
                model_choice = list(models.keys())[0]
                st.warning("best_model.txt not found — using first available model.")

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

    current_us_aqi = pm25_to_us_aqi(data['pm2_5'])
    cat, color, emoji = get_aqi_category(current_us_aqi)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("US AQI",      f"{current_us_aqi:.0f}")
    c2.metric("PM2.5",       f"{data['pm2_5']:.1f} μg/m³")
    c3.metric("Temperature", f"{data['temperature']:.1f}°C")
    c4.metric("Humidity",    f"{data['humidity']:.0f}%")

    st.markdown(f"""
    <div style='padding:1rem; background:{color}; color:black; border-radius:8px; margin:1rem 0;'>
        <h3>{emoji} Air Quality: {cat}</h3>
        <p>Current AQI for Karachi is {current_us_aqi:.0f} — {cat}</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.header("📅 3-Day AQI Forecast")

    if model_choice and models:
        st.info(f"🤖 Predictions by: **{model_choice}** (predicts PM2.5 → converts to US AQI)")
        preds = make_forecast(data, models[model_choice])
    else:
        st.warning("No model available for forecast.")
        return

    cols = st.columns(3)
    for i, pred in enumerate(preds):
        with cols[i]:
            st.subheader(pred['date'].strftime("%A"))
            st.metric(pred['date'].strftime("%b %d"), f"AQI {pred['aqi']:.0f}")
            st.caption(f"PM2.5: {pred['pm2_5']} μg/m³")
            cat2, _, em2 = get_aqi_category(pred["aqi"])
            st.markdown(f"{em2} **{cat2}**")

    st.markdown("---")
    st.subheader("📈 AQI Trend")

    dates  = ['Today'] + [p['date'].strftime("%a") for p in preds]
    values = [current_us_aqi] + [p['aqi'] for p in preds]

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

    if   current_us_aqi <= 50:  st.success("✅ Air quality is GOOD. Enjoy outdoor activities!")
    elif current_us_aqi <= 100: st.info("ℹ️ MODERATE. Sensitive people should be cautious.")
    elif current_us_aqi <= 150: st.warning("⚠️ UNHEALTHY for sensitive groups. Reduce outdoor activity.")
    elif current_us_aqi <= 200: st.error("🚨 UNHEALTHY. Everyone should limit outdoor exertion.")
    else:                       st.error("☠️ HAZARDOUS! Avoid all outdoor activities!")

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
