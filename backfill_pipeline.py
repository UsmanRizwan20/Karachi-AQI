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


def pm25_to_aqi(pm25):
    """
    Convert PM2.5 (µg/m³) to US EPA AQI (0–500 scale).
    This gives meaningful, varied AQI values for training.
    """
    breakpoints = [
        (0.0,   12.0,   0,   50),
        (12.1,  35.4,   51,  100),
        (35.5,  55.4,   101, 150),
        (55.5,  150.4,  151, 200),
        (150.5, 250.4,  201, 300),
        (250.5, 350.4,  301, 400),
        (350.5, 500.4,  401, 500),
    ]
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= pm25 <= c_high:
            return round(((i_high - i_low) / (c_high - c_low)) * (pm25 - c_low) + i_low, 1)
    return 500.0


def generate_historical_features(days=365):
    print("📡 Fetching current data from OpenWeatherMap API...")
    aqi, wx = fetch_current_data()

    comp = aqi["list"][0]["components"]

    base_pm25 = comp.get("pm2_5", 45.0)
    base_pm10 = comp.get("pm10", 80.0)
    base_no2  = comp.get("no2", 30.0)
    base_so2  = comp.get("so2", 15.0)
    base_o3   = comp.get("o3", 60.0)
    base_co   = comp.get("co", 500.0)

    temp  = wx["main"]["temp"]
    hum   = wx["main"]["humidity"]
    press = wx["main"]["pressure"]
    wind  = wx["wind"]["speed"]

    print(f"✔ Current readings: PM2.5={base_pm25:.1f}, Temp={temp:.1f}°C")

    rows = []
    now = datetime.now()
    np.random.seed(42)

    for i in range(days):
        ts = now - timedelta(days=days - i - 1)
        month = ts.month
        hour  = ts.hour

        # --- Seasonal variation (Karachi: worse in winter/monsoon) ---
        if month in [12, 1, 2]:      # winter - worst pollution
            season_factor = np.random.uniform(1.4, 2.2)
        elif month in [6, 7, 8]:     # monsoon - slightly better
            season_factor = np.random.uniform(0.6, 1.0)
        elif month in [3, 4, 5]:     # spring - moderate
            season_factor = np.random.uniform(0.9, 1.4)
        else:                        # fall - moderate-bad
            season_factor = np.random.uniform(1.1, 1.7)

        # --- Rush hour spikes ---
        if hour in [7, 8, 9, 17, 18, 19]:
            hour_factor = np.random.uniform(1.2, 1.6)
        elif hour in [0, 1, 2, 3, 4]:
            hour_factor = np.random.uniform(0.5, 0.8)
        else:
            hour_factor = np.random.uniform(0.9, 1.1)

        pm25 = round(base_pm25 * season_factor * hour_factor * np.random.uniform(0.8, 1.2), 2)
        pm10 = round(base_pm10 * season_factor * np.random.uniform(0.8, 1.2), 2)
        pm25 = max(5.0, pm25)
        pm10 = max(pm25, pm10)

        # Convert to real AQI (0-500 scale)
        aqi_value = pm25_to_aqi(pm25)

        temp_day = round(temp + np.random.uniform(-5, 5), 2)
        temp_min = round(temp_day - np.random.uniform(2, 5), 2)
        temp_max = round(temp_day + np.random.uniform(2, 5), 2)

        rows.append({
            "timestamp":    ts,
            "hour":         hour,
            "day":          ts.day,
            "month":        month,
            "day_of_week":  ts.weekday(),
            "season":       (month % 12 + 3) // 3,
            "is_weekend":   1 if ts.weekday() >= 5 else 0,
            "is_rush_hour": 1 if hour in [7, 8, 9, 17, 18, 19] else 0,
            "pm2_5":        pm25,
            "pm10":         pm10,
            "no2":          round(base_no2 * season_factor * np.random.uniform(0.6, 1.4), 2),
            "so2":          round(base_so2 * np.random.uniform(0.6, 1.4), 2),
            "o3":           round(base_o3  * np.random.uniform(0.7, 1.3), 2),
            "co":           round(base_co  * np.random.uniform(0.8, 1.2), 2),
            "pm_ratio":     round(pm25 / (pm10 + 1), 4),
            "temperature":  temp_day,
            "feels_like":   round(temp_day - np.random.uniform(1, 3), 2),
            "temp_min":     temp_min,
            "temp_max":     temp_max,
            "temp_range":   round(temp_max - temp_min, 2),
            "pressure":     int(press + np.random.uniform(-5, 5)),
            "humidity":     int(min(100, max(10, hum + np.random.uniform(-20, 20)))),
            "wind_speed":   round(max(0, wind + np.random.uniform(-2, 2)), 2),
            "wind_deg":     int(np.random.uniform(0, 360)),
            "clouds":       int(np.random.uniform(0, 100)),
            "hour_sin":     round(np.sin(2 * np.pi * hour / 24), 4),
            "hour_cos":     round(np.cos(2 * np.pi * hour / 24), 4),
            "aqi":          aqi_value,   # Real AQI (0-500 scale)
        })

    df = pd.DataFrame(rows)
    df = df.astype({
        "season":       "int64",
        "is_rush_hour": "int64",
        "pressure":     "int64",
        "humidity":     "int64",
        "wind_deg":     "int64",
        "clouds":       "int64",
    })

    print(f"✔ Generated {len(df)} records")
    print(f"✔ AQI range: {df['aqi'].min():.1f} – {df['aqi'].max():.1f} (mean: {df['aqi'].mean():.1f})")
    print(f"✔ AQI distribution:\n{df['aqi'].describe()}")
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
    print(f"✔ Successfully saved {len(df)} records to Hopsworks!")


if __name__ == "__main__":
    if not OPENWEATHER_API_KEY:
        sys.exit("❌ OPENWEATHERMAP_API_KEY missing")
    if not HOPSWORKS_API_KEY or not HOPSWORKS_PROJECT:
        sys.exit("❌ HOPSWORKS credentials missing")

    df = generate_historical_features(365)
    save_to_hopsworks(df)

