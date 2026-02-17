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


# ── AQI helpers ────────────────────────────────────────────────────────────────

def pm25_to_us_aqi(pm25: float) -> float:
    """
    Convert PM2.5 concentration (μg/m³) to US AQI using EPA breakpoints.
    Models now predict pm2_5 (continuous, high variance) instead of OWM's
    1-5 integer index (which was always 2 for Karachi → constant predictions).
    """
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


def get_aqi_category(aqi):
    if aqi <= 50:    return "Good", "#00E400", "😊"
    elif aqi <= 100: return "Moderate", "#FFFF00", "😐"
    elif aqi <= 150: return "Unhealthy for Sensitive Groups", "#FF7E00", "😷"
    elif aqi <= 200: return "Unhealthy", "#FF0000", "☹️"
    elif aqi <= 300: return "Very Unhealthy", "#8F3F97", "😨"
    else:            return "Hazardous", "#7E0023", "☠️"


# ── Data fetching ──────────────────────────────────────────────────────────────

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

        pm25 = c.get('pm2_5', 0)
        return {
            'pm2_5':       pm25,
            'pm10':        c.get('pm10', 0),
            'no2':         c.get('no2', 0),
            'o3':          c.get('o3', 0),
            'so2':         c.get('so2', 0),
            'co':          c.get('co', 0),
            'aqi_display': pm25_to_us_aqi(pm25),   # US AQI derived from live PM2.5
            'temperature': w['main']['temp'],
            'feels_like':  w['main']['feels_like'],
            'temp_min':    w['main']['temp_min'],
            'temp_max':    w['main']['temp_max'],
            'humidity':    w['main']['humidity'],
            'wind_speed':  w['wind']['speed'],
            'wind_deg':    w['wind'].get('deg', 0),
            'pressure':    w['main']['pressure'],
            'clouds':      w.get('clouds', {}).get('all', 0),
        }
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


@st.cache_data(ttl=3600)
def fetch_forecast_data():
    """
    Fetch real 5-day/3-hour weather + air pollution forecasts from OWM.
    Returns one representative entry per day for the next 3 days.
    Using real future API data (not synthetic offsets) ensures each day
    gets genuinely different pollutant/weather inputs for the model.
    """
    try:
        wx_r = requests.get("http://api.openweathermap.org/data/2.5/forecast",
            params={'lat': KARACHI_LAT, 'lon': KARACHI_LON,
                    'appid': OPENWEATHER_API_KEY, 'units': 'metric'}, timeout=10)
        wx_r.raise_for_status()
        wx_list = wx_r.json()['list']

        aqi_r = requests.get("http://api.openweathermap.org/data/2.5/air_pollution/forecast",
            params={'lat': KARACHI_LAT, 'lon': KARACHI_LON, 'appid': OPENWEATHER_API_KEY}, timeout=10)
        aqi_r.raise_for_status()
        aqi_list = aqi_r.json()['list']

        today = datetime.now().date()
        daily = {}
        for entry in wx_list:
            d = datetime.fromtimestamp(entry['dt']).date()
            if d <= today:
                continue
            daily.setdefault(d, []).append(entry)

        forecast_days = []
        for i, (date, entries) in enumerate(sorted(daily.items())):
            if i >= 3:
                break
            best = min(entries, key=lambda e: abs(datetime.fromtimestamp(e['dt']).hour - 12))
            ts   = best['dt']
            dt   = datetime.fromtimestamp(ts)
            closest_aqi = min(aqi_list, key=lambda e: abs(e['dt'] - ts))
            comp = closest_aqi['components']
            pm25 = comp.get('pm2_5', 0)

            forecast_days.append({
                'date':        dt,
                'pm2_5':       pm25,
                'pm10':        comp.get('pm10', 0),
                'no2':         comp.get('no2', 0),
                'o3':          comp.get('o3', 0),
                'so2':         comp.get('so2', 0),
                'co':          comp.get('co', 0),
                'aqi_display': pm25_to_us_aqi(pm25),  # OWM's own forecast AQI (fallback)
                'temperature': best['main']['temp'],
                'feels_like':  best['main']['feels_like'],
                'temp_min':    best['main']['temp_min'],
                'temp_max':    best['main']['temp_max'],
                'humidity':    best['main']['humidity'],
                'wind_speed':  best['wind']['speed'],
                'wind_deg':    best['wind'].get('deg', 0),
                'pressure':    best['main']['pressure'],
                'clouds':      best.get('clouds', {}).get('all', 0),
            })

        return forecast_days

    except Exception as e:
        st.warning(f"Forecast API error: {e}")
        return None


# ── Model loading ──────────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    models = {}
    for name, path in [
        ('Random Forest', MODEL_DIR / "random_forest_model.pkl"),
        ('XGBoost',       MODEL_DIR / "xgboost_model.pkl"),
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
    path = MODEL_DIR / "feature_names.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ── Prediction ─────────────────────────────────────────────────────────────────

def build_feature_row(day_data: dict, feature_names: list) -> pd.DataFrame:
    """
    Build a feature row matching exactly the columns the model was trained on.
    Uses real API data for each forecast day, so inputs differ meaningfully day-to-day.
    """
    dt           = day_data['date']
    hour         = dt.hour
    day          = dt.day
    month        = dt.month
    day_of_week  = dt.weekday()
    is_weekend   = int(day_of_week >= 5)
    is_rush_hour = int(hour in range(7, 10) or hour in range(17, 20))
    season       = (month % 12) // 3

    temperature = day_data['temperature']
    feels_like  = day_data.get('feels_like', temperature - 2.0)
    temp_min    = day_data.get('temp_min',   temperature - 3.0)
    temp_max    = day_data.get('temp_max',   temperature + 3.0)
    temp_range  = temp_max - temp_min

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    pm2_5    = day_data['pm2_5']
    pm10     = day_data['pm10']
    # feature_pipeline uses pm2_5 / (pm10 + 1) to avoid division by zero
    pm_ratio = pm2_5 / (pm10 + 1)

    all_values = {
        'hour':           hour,
        'day':            day,
        'month':          month,
        'day_of_week':    day_of_week,
        'is_weekend':     is_weekend,
        'season':         season,
        'pm2_5':          pm2_5,
        'pm10':           pm10,
        'no2':            day_data['no2'],
        'o3':             day_data['o3'],
        'so2':            day_data['so2'],
        'co':             day_data['co'],
        'temperature':    temperature,
        'feels_like':     feels_like,
        'temp_min':       temp_min,
        'temp_max':       temp_max,
        'pressure':       day_data['pressure'],
        'humidity':       day_data['humidity'],
        'wind_speed':     day_data['wind_speed'],
        'wind_deg':       day_data.get('wind_deg', 0.0),
        'clouds':         day_data.get('clouds', 0.0),
        'pm_ratio':       pm_ratio,
        'temp_range':     temp_range,
        'is_rush_hour':   is_rush_hour,
        'hour_sin':       hour_sin,
        'hour_cos':       hour_cos,
        'aqi_change_rate': 0.0,   # unknown at inference time; use neutral value
    }

    # Select only the features the model was actually trained on, in the right order
    row = {k: all_values[k] for k in feature_names if k in all_values}
    return pd.DataFrame([row])[feature_names]


def predict_pm25(model_obj, feature_df) -> float:
    """
    Run prediction. Models now predict PM2.5 (μg/m³), not OWM's 1-5 index.
    Caller converts the result to US AQI via pm25_to_us_aqi().
    """
    if isinstance(model_obj, dict):
        scaled = model_obj['scaler'].transform(feature_df)
        raw    = float(model_obj['model'].predict(scaled)[0])
    else:
        raw = float(model_obj.predict(feature_df)[0])
    return max(0.0, raw)   # PM2.5 can't be negative


def make_forecast(forecast_days, model_obj, feature_names):
    preds = []
    for day_data in forecast_days:
        feature_df = build_feature_row(day_data, feature_names)
        try:
            predicted_pm25 = predict_pm25(model_obj, feature_df)
            predicted_aqi  = pm25_to_us_aqi(predicted_pm25)
        except Exception as e:
            st.warning(f"Prediction error for {day_data['date'].strftime('%A')}: {e}")
            # Fall back to OWM's own forecast converted to US AQI
            predicted_aqi  = day_data['aqi_display']
            predicted_pm25 = day_data['pm2_5']
        preds.append({
            'date':        day_data['date'],
            'pm2_5':       round(predicted_pm25, 1),
            'aqi_display': predicted_aqi,
        })
    return preds


# ── Main app ───────────────────────────────────────────────────────────────────

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

    with st.spinner("Fetching live AQI data..."):
        data = fetch_current_aqi()

    if not data:
        st.error("Cannot fetch data. Check OPENWEATHERMAP_API_KEY in .env file.")
        return

    # ── Current AQI ────────────────────────────────────────────────────────────
    st.header("📊 Current Air Quality")

    aqi_display = data['aqi_display']
    cat, color, emoji = get_aqi_category(aqi_display)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AQI (US Scale)", f"{aqi_display:.0f}")
    c2.metric("PM2.5",          f"{data['pm2_5']:.1f} μg/m³")
    c3.metric("Temperature",    f"{data['temperature']:.1f}°C")
    c4.metric("Humidity",       f"{data['humidity']:.0f}%")

    st.markdown(f"""
    <div style='padding:1rem; background:{color}; color:black; border-radius:8px; margin:1rem 0;'>
        <h3>{emoji} Air Quality: {cat}</h3>
        <p>Current AQI for {CITY_NAME} is {aqi_display:.0f} — {cat}</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── 3-Day Forecast ──────────────────────────────────────────────────────────
    st.header("📅 3-Day AQI Forecast")

    feature_names = load_feature_names()
    if not feature_names:
        st.error("feature_names.json not found. Please re-run training_pipeline.py.")
        return

    with st.spinner("Fetching forecast data..."):
        forecast_days = fetch_forecast_data()

    if not forecast_days:
        st.warning("Could not fetch forecast data from OpenWeatherMap. Try refreshing.")
        return

    if model_choice and models:
        st.info(f"🤖 Predictions by: **{model_choice}** — predicting PM2.5 → converted to US AQI")
        preds = make_forecast(forecast_days, models[model_choice], feature_names)
    else:
        st.warning("No model available for forecast.")
        return

    cols = st.columns(3)
    for i, pred in enumerate(preds):
        with cols[i]:
            st.subheader(pred['date'].strftime("%A"))
            st.metric(pred['date'].strftime("%b %d"), f"AQI {pred['aqi_display']:.0f}")
            st.caption(f"PM2.5: {pred['pm2_5']} μg/m³")
            cat2, _, em2 = get_aqi_category(pred["aqi_display"])
            st.markdown(f"{em2} **{cat2}**")

    st.markdown("---")
    st.subheader("📈 AQI Trend")

    dates  = ['Today'] + [p['date'].strftime("%a") for p in preds]
    values = [aqi_display] + [p['aqi_display'] for p in preds]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=values,
        mode='lines+markers',
        line=dict(color='#00E400', width=3),
        marker=dict(size=12)
    ))
    fig.update_layout(
        title=f'AQI Forecast - {CITY_NAME}',
        xaxis_title='Day', yaxis_title='AQI (US Scale)',
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
    st.markdown(f"""
    <div style='text-align:center; color:#888; padding:2rem;'>
        <p>🤖 Models: Random Forest | XGBoost | Neural Network (MLP)</p>
        <p>📡 Live data: OpenWeatherMap API | 🗄️ Feature Store: Hopsworks</p>
        <p>🌍 {CITY_NAME} AQI Predictor — Serverless ML Pipeline</p>
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    if not OPENWEATHER_API_KEY:
        st.error("❌ OPENWEATHERMAP_API_KEY not found in .env file!")
    else:
        main()
