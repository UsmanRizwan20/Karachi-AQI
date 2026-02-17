import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────
OPENWEATHER_API_KEY   = os.getenv("OPENWEATHERMAP_API_KEY")
HOPSWORKS_API_KEY     = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT     = os.getenv("HOPSWORKS_PROJECT_NAME")

KARACHI_LAT           = float(os.getenv("CITY_LAT", "24.8607"))
KARACHI_LON           = float(os.getenv("CITY_LON", "67.0011"))
CITY_NAME             = os.getenv("CITY_NAME", "Karachi")

FEATURE_GROUP_NAME    = os.getenv("FEATURE_GROUP_NAME", "karachi_aqi_features")
FEATURE_GROUP_VERSION = int(os.getenv("FEATURE_GROUP_VERSION", "1"))

# Local cache used to compute AQI change rate between pipeline runs
AQI_CACHE_PATH = Path("data/last_aqi.txt")


def fetch_air_quality_data(lat: float, lon: float) -> Optional[Dict]:
    """Fetch current air quality data from OpenWeatherMap API."""
    try:
        response = requests.get(
            "http://api.openweathermap.org/data/2.5/air_pollution",
            params={"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY},
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching air quality data: {e}")
        return None


def fetch_weather_data(lat: float, lon: float) -> Optional[Dict]:
    """Fetch current weather data from OpenWeatherMap API."""
    try:
        response = requests.get(
            "http://api.openweathermap.org/data/2.5/weather",
            params={"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY, "units": "metric"},
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return None


def load_last_aqi() -> Optional[float]:
    """
    Load the previous AQI value from local cache.
    Used to compute the AQI change rate feature between pipeline runs.
    Returns None on the very first run.
    """
    try:
        if AQI_CACHE_PATH.exists():
            return float(AQI_CACHE_PATH.read_text().strip())
    except Exception:
        pass
    return None


def save_last_aqi(aqi_value: float):
    """Persist the current AQI to disk so the next hourly run can compute change rate."""
    AQI_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    AQI_CACHE_PATH.write_text(str(aqi_value))


def compute_features(aqi_data: Dict, weather_data: Dict) -> pd.DataFrame:
    """
    Compute features from raw API data.

    Features:
      Time-based  : hour, day, month, day_of_week, season, is_weekend, is_rush_hour
      Cyclical    : hour_sin, hour_cos
      Pollutants  : pm2_5, pm10, no2, o3, so2, co
      Weather     : temperature, feels_like, temp_min, temp_max, pressure,
                    humidity, wind_speed, wind_deg, clouds
      Derived     : pm_ratio, temp_range, aqi_change_rate
      Target      : aqi
    """
    timestamp   = datetime.fromtimestamp(aqi_data["list"][0]["dt"])
    components  = aqi_data["list"][0]["components"]
    current_aqi = float(aqi_data["list"][0]["main"]["aqi"])

    # ── Time features ─────────────────────────────────────────────────────────
    features = {
        "timestamp":    timestamp,
        "hour":         timestamp.hour,
        "day":          timestamp.day,
        "month":        timestamp.month,
        "day_of_week":  timestamp.weekday(),
        "is_weekend":   1 if timestamp.weekday() >= 5 else 0,
        "season":       (timestamp.month % 12 + 3) // 3,  # 1=Winter 2=Spring 3=Summer 4=Fall
        "is_rush_hour": 1 if timestamp.hour in [7, 8, 9, 17, 18, 19] else 0,
        # Cyclical encoding keeps hour continuity (23→0 wraps smoothly)
        "hour_sin":     np.sin(2 * np.pi * timestamp.hour / 24),
        "hour_cos":     np.cos(2 * np.pi * timestamp.hour / 24),
    }

    # ── Pollutant features ────────────────────────────────────────────────────
    features.update({
        "pm2_5": components.get("pm2_5", 0),
        "pm10":  components.get("pm10",  0),
        "no2":   components.get("no2",   0),
        "o3":    components.get("o3",    0),
        "so2":   components.get("so2",   0),
        "co":    components.get("co",    0),
        "aqi":   current_aqi,
    })

    # ── Weather features ──────────────────────────────────────────────────────
    features.update({
        "temperature": weather_data["main"]["temp"],
        "feels_like":  weather_data["main"]["feels_like"],
        "temp_min":    weather_data["main"]["temp_min"],
        "temp_max":    weather_data["main"]["temp_max"],
        "pressure":    weather_data["main"]["pressure"],
        "humidity":    weather_data["main"]["humidity"],
        "wind_speed":  weather_data["wind"]["speed"],
        "wind_deg":    weather_data["wind"].get("deg", 0),
        "clouds":      weather_data["clouds"]["all"],
    })

    # ── Derived features ──────────────────────────────────────────────────────
    features["pm_ratio"]   = features["pm2_5"] / (features["pm10"] + 1)
    features["temp_range"] = features["temp_max"] - features["temp_min"]

    # AQI change rate: how much AQI changed since the last hourly run
    # On the very first run this is 0.0; after that it tracks real change
    last_aqi = load_last_aqi()
    features["aqi_change_rate"] = float(current_aqi - last_aqi) if last_aqi is not None else 0.0
    save_last_aqi(current_aqi)

    return pd.DataFrame([features])


def save_to_hopsworks(df: pd.DataFrame) -> bool:
    """Save features to Hopsworks Feature Store, with local CSV fallback."""
    try:
        import hopsworks

        print(f"\nConnecting to Hopsworks project: {HOPSWORKS_PROJECT}")
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT)
        fs      = project.get_feature_store()
        print("Connected to Feature Store!")

        feature_group = fs.get_or_create_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION,
            description=f"AQI and weather features for {CITY_NAME}",
            primary_key=["timestamp"],
            event_time="timestamp",
        )
        print(f"Feature group: {FEATURE_GROUP_NAME} v{FEATURE_GROUP_VERSION}")
        feature_group.insert(df, write_options={"wait_for_job": False})
        print(f"✅ Successfully saved {len(df)} records to Hopsworks!")
        return True

    except Exception as e:
        print(f"❌ Error saving to Hopsworks: {e}")
        local_path = Path("data/features")
        local_path.mkdir(parents=True, exist_ok=True)
        filepath = local_path / f"features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filepath, index=False)
        print(f"💾 Saved locally to: {filepath}")
        return False


def run_feature_pipeline():
    """Main entry point for the hourly feature pipeline."""
    print("=" * 70)
    print(f"🚀 {CITY_NAME.upper()} AQI FEATURE PIPELINE")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    print("\n📡 Step 1: Fetching air quality data...")
    aqi_data = fetch_air_quality_data(KARACHI_LAT, KARACHI_LON)
    if not aqi_data:
        print("❌ Failed to fetch air quality data. Exiting.")
        return False
    print("✅ Air quality data fetched!")

    print("\n🌤️  Step 2: Fetching weather data...")
    weather_data = fetch_weather_data(KARACHI_LAT, KARACHI_LON)
    if not weather_data:
        print("❌ Failed to fetch weather data. Exiting.")
        return False
    print("✅ Weather data fetched!")

    print("\n🔧 Step 3: Computing features (including AQI change rate)...")
    features_df = compute_features(aqi_data, weather_data)
    print(f"✅ Computed {len(features_df.columns)} features")
    print(f"   AQI:             {features_df['aqi'].values[0]}")
    print(f"   AQI Change Rate: {features_df['aqi_change_rate'].values[0]:+.2f}  ← new feature")
    print(f"   PM2.5:           {features_df['pm2_5'].values[0]:.2f} μg/m³")
    print(f"   Temperature:     {features_df['temperature'].values[0]:.1f}°C")

    print("\n💾 Step 4: Saving to Feature Store...")
    success = save_to_hopsworks(features_df)
    print("\n✅ Feature pipeline completed!" if success else "\n⚠️  Completed with warnings (saved locally)")
    print("=" * 70)
    return success


if __name__ == "__main__":
    if not OPENWEATHER_API_KEY:
        print("❌ OPENWEATHERMAP_API_KEY not found in .env")
        sys.exit(1)
    if not HOPSWORKS_API_KEY or not HOPSWORKS_PROJECT:
        print("⚠️  Hopsworks credentials not found — will save locally")
    run_feature_pipeline()