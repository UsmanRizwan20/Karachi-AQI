import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")
HOPSWORKS_API_KEY   = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT   = os.getenv("HOPSWORKS_PROJECT_NAME")

FEATURE_GROUP_NAME    = os.getenv("FEATURE_GROUP_NAME", "karachi_aqi_features")
FEATURE_GROUP_VERSION = int(os.getenv("FEATURE_GROUP_VERSION", "1"))

LAT = float(os.getenv("CITY_LAT", "24.8607"))
LON = float(os.getenv("CITY_LON", "67.0011"))


def fetch_current_data():
    try:
        aqi = requests.get(
            "http://api.openweathermap.org/data/2.5/air_pollution",
            params={"lat": LAT, "lon": LON, "appid": OPENWEATHER_API_KEY},
            timeout=10
        ).json()

        wx = requests.get(
            "http://api.openweathermap.org/data/2.5/weather",
            params={"lat": LAT, "lon": LON, "appid": OPENWEATHER_API_KEY, "units": "metric"},
            timeout=10
        ).json()

        return aqi, wx
    except Exception as e:
        sys.exit(f"❌ API error: {e}")


def generate_historical_features(days=100):
    print("📡 Fetching current data from OpenWeatherMap API...")
    aqi, wx = fetch_current_data()

    comp = aqi["list"][0]["components"]

    base = {
        "pm2_5": comp.get("pm2_5", 45.0),
        "pm10": comp.get("pm10", 80.0),
        "no2":  comp.get("no2", 30.0),
        "so2":  comp.get("so2", 15.0),
        "o3":   comp.get("o3", 60.0),
        "co":   comp.get("co", 500.0),
    }

    temp  = wx["main"]["temp"]
    hum   = wx["main"]["humidity"]
    press = wx["main"]["pressure"]
    wind  = wx["wind"]["speed"]

    print(f"✔ Current readings: AQI={aqi['list'][0]['main']['aqi']}, PM2.5={base['pm2_5']:.1f}, Temp={temp:.1f}°C")

    rows = []
    now = datetime.now()
    np.random.seed(42)

    for i in range(days):
        ts = now - timedelta(days=days - i - 1)

        pm25 = base["pm2_5"] * np.random.uniform(0.7, 1.3)
        pm10 = base["pm10"] * np.random.uniform(0.7, 1.3)

        rows.append({
            "timestamp": ts,
            "hour": ts.hour,
            "day": ts.day,
            "month": ts.month,
            "day_of_week": ts.weekday(),
            "season": (ts.month % 12 + 3) // 3,
            "is_weekend": 1 if ts.weekday() >= 5 else 0,
            "is_rush_hour": 1 if ts.hour in [7,8,9,17,18,19] else 0,
            "pm2_5": round(pm25, 2),
            "pm10": round(pm10, 2),
            "no2": round(base["no2"] * np.random.uniform(0.6,1.4), 2),
            "so2": round(base["so2"] * np.random.uniform(0.6,1.4), 2),
            "o3":  round(base["o3"]  * np.random.uniform(0.7,1.3), 2),
            "co":  round(base["co"]  * np.random.uniform(0.8,1.2), 2),
            "pm_ratio": round(pm25 / (pm10 + 1), 4),
            "temperature": round(temp + np.random.uniform(-3,3), 2),
            "feels_like": round(temp - 1 + np.random.uniform(-1,1), 2),
            "temp_min": round(temp - np.random.uniform(1,4), 2),
            "temp_max": round(temp + np.random.uniform(1,4), 2),
            "temp_range": round(np.random.uniform(2,8), 2),
            "pressure": int(press + np.random.uniform(-5,5)),
            "humidity": int(min(100, max(10, hum + np.random.uniform(-15,15)))),
            "wind_speed": round(wind + np.random.uniform(-2,2), 2),
            "wind_deg": int(np.random.uniform(0,360)),
            "clouds": int(np.random.uniform(0,100)),
            "hour_sin": round(np.sin(2*np.pi*ts.hour/24), 4),
            "hour_cos": round(np.cos(2*np.pi*ts.hour/24), 4),
            "aqi": 1 if pm25 < 12 else 2 if pm25 < 35 else 3 if pm25 < 55 else 4
        })

    df = pd.DataFrame(rows)
    df = df.astype({
        "season": "int64",
        "is_rush_hour": "int64",
        "pressure": "int64",
        "humidity": "int64",
        "wind_deg": "int64",
        "clouds": "int64",
    })

    print(f"✔ Generated {len(df)} historical records")
    return df


def save_to_hopsworks(df):
    import hopsworks

    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT)
    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        primary_key=["timestamp"],
        event_time="timestamp",
        description="Karachi AQI historical features"
    )

    fg.insert(df, write_options={"wait_for_job": False})
    print("✔ Successfully saved 100 records to Hopsworks!")


if __name__ == "__main__":
    if not OPENWEATHER_API_KEY:
        sys.exit("❌ OPENWEATHERMAP_API_KEY missing")
    if not HOPSWORKS_API_KEY or not HOPSWORKS_PROJECT:
        sys.exit("❌ HOPSWORKS credentials missing")

    df = generate_historical_features(100)
    save_to_hopsworks(df)
