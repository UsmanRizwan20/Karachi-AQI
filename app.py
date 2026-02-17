import os
from pathlib import Path
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
import joblib
import torch
import torch.nn as nn
from dotenv import load_dotenv

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")
KARACHI_LAT = float(os.getenv("CITY_LAT", "24.8607"))
KARACHI_LON = float(os.getenv("CITY_LON", "67.0011"))
CITY_NAME = os.getenv("CITY_NAME", "Karachi")
MODEL_DIR = Path("models")

st.set_page_config(page_title=f"{CITY_NAME} AQI Predictor",
                   page_icon="🌍", layout="wide")

st.markdown("""
<style>
    h1 { font-weight: 700; }
    .stMetric { background-color: #1E2433; padding: 1.2rem; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


def get_aqi_category(aqi_1_to_5):
    display = aqi_1_to_5 * 50
    if display <= 50:
        return "Good", "#00E400", "😊"
    elif display <= 100:
        return "Moderate", "#FFFF00", "😐"
    elif display <= 150:
        return "Unhealthy for Sensitive Groups", "#FF7E00", "😷"
    elif display <= 200:
        return "Unhealthy", "#FF0000", "☹️"
    elif display <= 250:
        return "Very Unhealthy", "#8F3F97", "😨"
    else:
        return "Hazardous", "#7E0023", "☠️"


@st.cache_data(ttl=3600)
def fetch_current_data():
    try:
        aqi_r = requests.get(
            "http://api.openweathermap.org/data/2.5/air_pollution",
            params={"lat": KARACHI_LAT, "lon": KARACHI_LON, "appid": OPENWEATHER_API_KEY},
            timeout=10,
        )
        aqi_r.raise_for_status()

        wx_r = requests.get(
            "http://api.openweathermap.org/data/2.5/weather",
            params={"lat": KARACHI_LAT, "lon": KARACHI_LON,
                    "appid": OPENWEATHER_API_KEY, "units": "metric"},
            timeout=10,
        )
        wx_r.raise_for_status()

        c = aqi_r.json()["list"][0]["components"]
        w = wx_r.json()

        return {
            "aqi": float(aqi_r.json()["list"][0]["main"]["aqi"]),
            "pm2_5": c.get("pm2_5", 0.0),
            "pm10": c.get("pm10", 0.0),
            "no2": c.get("no2", 0.0),
            "o3": c.get("o3", 0.0),
            "so2": c.get("so2", 0.0),
            "co": c.get("co", 0.0),
            "temperature": w["main"]["temp"],
            "feels_like": w["main"]["feels_like"],
            "temp_min": w["main"]["temp_min"],
            "temp_max": w["main"]["temp_max"],
            "pressure": w["main"]["pressure"],
            "humidity": w["main"]["humidity"],
            "wind_speed": w["wind"]["speed"],
            "wind_deg": w["wind"].get("deg", 0),
            "clouds": w["clouds"]["all"],
        }
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


def build_feature_vector(data, ts, prev_aqi=None):
    pm2_5 = data["pm2_5"]
    pm10 = data["pm10"]

    row = {
        "hour": ts.hour,
        "day": ts.day,
        "month": ts.month,
        "day_of_week": ts.weekday(),
        "is_weekend": 1 if ts.weekday() >= 5 else 0,
        "season": (ts.month % 12 + 3) // 3,
        "is_rush_hour": 1 if ts.hour in [7, 8, 9, 17, 18, 19] else 0,
        "hour_sin": np.sin(2 * np.pi * ts.hour / 24),
        "hour_cos": np.cos(2 * np.pi * ts.hour / 24),
        "pm2_5": pm2_5,
        "pm10": pm10,
        "no2": data["no2"],
        "o3": data["o3"],
        "so2": data["so2"],
        "co": data["co"],
        "temperature": data["temperature"],
        "feels_like": data["feels_like"],
        "temp_min": data["temp_min"],
        "temp_max": data["temp_max"],
        "pressure": data["pressure"],
        "humidity": data["humidity"],
        "wind_speed": data["wind_speed"],
        "wind_deg": data["wind_deg"],
        "clouds": data["clouds"],
        "pm_ratio": pm2_5 / (pm10 + 1),
        "temp_range": data["temp_max"] - data["temp_min"],
        "aqi_change_rate": float(data["aqi"] - prev_aqi) if prev_aqi is not None else 0.0,
    }
    return pd.DataFrame([row])


class AQINet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


@st.cache_resource
def load_models():
    models = {}
    scaler = None

    scaler_path = MODEL_DIR / "scaler.pkl"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)

    for name, path in [
        ("Random Forest", MODEL_DIR / "random_forest_model.pkl"),
        ("XGBoost", MODEL_DIR / "xgboost_model.pkl"),
        ("Ridge Regression", MODEL_DIR / "ridge_model.pkl"),
    ]:
        if path.exists():
            models[name] = joblib.load(path)

    pt_path = MODEL_DIR / "pytorch_dnn.pt"
    if pt_path.exists():
        ckpt = torch.load(pt_path, map_location="cpu")
        net = AQINet(ckpt["input_dim"])
        net.load_state_dict(ckpt["model_state"])
        net.eval()
        models["PyTorch DNN"] = net

    return models, scaler


def predict_aqi(model, model_name, features_df, scaler):
    try:
        if model_name == "PyTorch DNN":
            x = torch.tensor(
                scaler.transform(features_df) if scaler else features_df.values,
                dtype=torch.float32,
            )
            with torch.no_grad():
                pred = model(x).item()
        elif model_name == "Ridge Regression":
            X = scaler.transform(features_df) if scaler else features_df.values
            pred = float(model.predict(X)[0])
        else:
            pred = float(model.predict(features_df)[0])

        return float(np.clip(round(pred * 2) / 2, 1.0, 5.0))
    except Exception as e:
        st.warning(f"Prediction error ({model_name}): {e}")
        return None


def make_real_forecast(data, model, model_name, scaler, days=3):
    preds = []
    prev_aqi = data["aqi"]

    for i in range(1, days + 1):
        future_ts = datetime.now() + timedelta(days=i)
        fv = build_feature_vector(data, future_ts, prev_aqi)
        aqi_pred = predict_aqi(model, model_name, fv, scaler)
        if aqi_pred is None:
            aqi_pred = prev_aqi
        preds.append({
            "date": future_ts,
            "aqi": aqi_pred,
            "features": fv,
        })
        prev_aqi = aqi_pred

    return preds


def main():
    st.title(f"🌍 {CITY_NAME} Air Quality Index Predictor")
    st.markdown("### Real-time AQI Monitoring & 3-Day ML Forecast")
    st.markdown("---")

    with st.sidebar:
        st.header("⚙️ Settings")
        models, scaler = load_models()

        st.markdown("### 🤖 Models Available")
        if models:
            for m in models:
                st.write(f"✅ {m}")
        else:
            st.warning("No trained models found — run training_pipeline.py")

        best_txt = MODEL_DIR / "best_model.txt"
        if best_txt.exists() and best_txt.read_text().strip() in models:
            model_choice = best_txt.read_text().strip()
            st.success(f"🏆 Auto-selected: **{model_choice}**")
        elif models:
            model_choice = list(models.keys())[0]
            st.info(f"Using: {model_choice}")
        else:
            model_choice = None

        st.markdown("---")
        st.markdown("### 📍 Location")
        st.write(f"**City:** {CITY_NAME}")
        st.write(f"**Lat:** {KARACHI_LAT}  |  **Lon:** {KARACHI_LON}")

        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()

    with st.spinner("Fetching live data ..."):
        data = fetch_current_data()
    if not data:
        st.error("Cannot fetch data. Check OPENWEATHERMAP_API_KEY.")
        return

    st.header("📊 Current Air Quality")

    cat, color, emoji = get_aqi_category(data["aqi"])

    if data["aqi"] >= 4:
        st.error(f"🚨 HAZARDOUS AIR QUALITY ALERT — {cat}! Avoid all outdoor activity.")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("AQI (1-5)", f"{data['aqi']:.0f}")
    c2.metric("PM2.5", f"{data['pm2_5']:.1f} μg/m³")
    c3.metric("PM10", f"{data['pm10']:.1f} μg/m³")
    c4.metric("Temperature", f"{data['temperature']:.1f} °C")
    c5.metric("Humidity", f"{data['humidity']} %")

    st.markdown(f"""
    <div style='padding:1rem; background:{color}; color:black; border-radius:8px; margin:1rem 0;'>
        <h3>{emoji} Air Quality: {cat}</h3>
        <p>Current AQI for {CITY_NAME} is {data['aqi']:.0f} — {cat}</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    st.header("📅 3-Day AQI Forecast  (ML-powered)")

    if model_choice and models:
        model = models[model_choice]
        st.info(f"🤖 Predictions from **{model_choice}**")
        preds = make_real_forecast(data, model, model_choice, scaler)
    else:
        st.warning("No model available — train models first.")
        preds = [{"date": datetime.now() + timedelta(days=i),
                  "aqi": data["aqi"]} for i in range(1, 4)]

    cols = st.columns(3)
    for i, pred in enumerate(preds):
        with cols[i]:
            st.subheader(pred["date"].strftime("%A"))
            cat2, col2, em2 = get_aqi_category(pred["aqi"])
            st.metric(pred["date"].strftime("%b %d"), f"AQI {pred['aqi']:.1f}")
            st.markdown(
                f"<span style='color:{col2}; font-weight:bold;'>{em2} {cat2}</span>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    st.subheader("📈 AQI Trend (Today + 3-Day Forecast)")

    dates = ["Today"] + [p["date"].strftime("%a %b %d") for p in preds]
    values = [data["aqi"]] + [p["aqi"] for p in preds]
    colors = [get_aqi_category(v)[1] for v in values]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=values,
        mode="lines+markers",
        line=dict(color="#00E400", width=3),
        marker=dict(size=14, color=colors, line=dict(width=2, color="white")),
    ))
    fig.update_layout(
        title=f"AQI Forecast — {CITY_NAME}",
        xaxis_title="Day", yaxis_title="AQI (1–5)",
        height=400,
        plot_bgcolor="#1A1F2E", paper_bgcolor="#1A1F2E",
        font=dict(color="white"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.header("🔬 Pollutant Breakdown")

    fig2 = go.Figure(data=[go.Bar(
        x=["PM2.5", "PM10", "NO₂", "O₃", "SO₂", "CO"],
        y=[data["pm2_5"], data["pm10"], data["no2"],
           data["o3"], data["so2"], data["co"]],
    )])
    fig2.update_layout(
        title="Current Pollutant Levels (μg/m³)",
        height=400,
        plot_bgcolor="#1A1F2E", paper_bgcolor="#1A1F2E",
        font=dict(color="white"),
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    st.header("🔍 SHAP Feature Importance")

    shap_img = MODEL_DIR / "shap_summary.png"
    shap_csv = MODEL_DIR / "shap_importance.csv"

    if shap_img.exists():
        st.image(str(shap_img), caption=f"SHAP Summary — {model_choice}",
                 use_container_width=True)
    elif shap_csv.exists():
        shap_df = pd.read_csv(shap_csv).head(10)
        fig3 = px.bar(
            shap_df, x="shap_mean", y="feature",
            orientation="h", color="shap_mean",
            color_continuous_scale="Viridis",
            title="Top 10 Features by Mean |SHAP| Value",
        )
        fig3.update_layout(
            height=400, yaxis=dict(autorange="reversed"),
            plot_bgcolor="#1A1F2E", paper_bgcolor="#1A1F2E",
            font=dict(color="white"),
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Run training_pipeline.py to generate SHAP explanations.")

    st.markdown("---")

    st.header("💡 Health Recommendations")

    aqi_display = data["aqi"] * 50
    if aqi_display <= 50:
        st.success("✅ GOOD — Enjoy outdoor activities!")
    elif aqi_display <= 100:
        st.info("ℹ️ MODERATE — Sensitive people should be cautious.")
    elif aqi_display <= 150:
        st.warning("⚠️ UNHEALTHY for Sensitive Groups — Reduce outdoor activity.")
    elif aqi_display <= 200:
        st.error("🚨 UNHEALTHY — Everyone should limit outdoor exertion.")
    else:
        st.error("☠️ HAZARDOUS — Avoid ALL outdoor activities immediately!")


if __name__ == "__main__":
    if not OPENWEATHER_API_KEY:
        st.error("OPENWEATHERMAP_API_KEY not found in .env")
    else:
        main()
