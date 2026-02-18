

import os
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
OPENWEATHER_API_KEY   = os.getenv("OPENWEATHERMAP_API_KEY")
HOPSWORKS_API_KEY     = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT     = os.getenv("HOPSWORKS_PROJECT_NAME")
FEATURE_GROUP_NAME    = os.getenv("FEATURE_GROUP_NAME", "karachi_aqi_features")
FEATURE_GROUP_VERSION = int(os.getenv("FEATURE_GROUP_VERSION", "1"))

LAT  = float(os.getenv("CITY_LAT",  "24.8607"))
LON  = float(os.getenv("CITY_LON",  "67.0011"))
CITY = os.getenv("CITY_NAME", "Karachi")

# Set to True if you have a paid OWM subscription (enables weather history)
USE_PAID_WEATHER = os.getenv("USE_PAID_WEATHER", "false").lower() == "true"

DAYS_TO_FETCH = int(os.getenv("BACKFILL_DAYS", "100"))


# ── Helpers ───────────────────────────────────────────────────────────────────

def pm25_to_us_aqi(pm25: float) -> float:
    """EPA PM2.5 → US AQI linear interpolation."""
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


def karachi_seasonal_temp(month: int, hour: int) -> dict:
    """
    Realistic Karachi monthly temperature and humidity normals.
    Used as fallback when paid weather history is not available.
    Source: Pakistan Meteorological Department climatological averages.
    """
    # (avg_temp_c, daily_range, avg_humidity_pct, avg_wind_ms, avg_pressure_hpa)
    monthly = {
        1:  (19.0, 8.0, 68, 3.5, 1016),
        2:  (20.5, 9.0, 64, 4.0, 1014),
        3:  (24.5, 9.0, 62, 4.5, 1012),
        4:  (28.5, 8.0, 64, 5.0, 1009),
        5:  (31.5, 7.0, 67, 5.5, 1006),
        6:  (32.0, 6.0, 73, 6.0, 1003),
        7:  (30.5, 5.0, 80, 5.5, 1003),
        8:  (29.5, 5.0, 80, 5.0, 1004),
        9:  (29.0, 6.0, 76, 4.5, 1006),
        10: (27.5, 7.0, 68, 4.0, 1010),
        11: (24.0, 8.0, 62, 3.5, 1014),
        12: (20.5, 8.0, 65, 3.5, 1016),
    }
    avg_temp, daily_range, avg_hum, avg_wind, avg_press = monthly[month]

    # Diurnal: coldest ~5am, hottest ~3pm
    hour_offset = -daily_range / 2 * np.cos(2 * np.pi * (hour - 15) / 24)
    temp = avg_temp + hour_offset

    # Wind picks up midday
    wind_factor = 0.7 + 0.6 * np.sin(np.pi * hour / 24)
    wind = avg_wind * wind_factor

    return {
        "temperature": round(temp, 1),
        "feels_like":  round(temp - 1.5, 1),
        "temp_min":    round(avg_temp - daily_range / 2, 1),
        "temp_max":    round(avg_temp + daily_range / 2, 1),
        "temp_range":  round(daily_range, 1),
        "humidity":    int(avg_hum),
        "wind_speed":  round(wind, 2),
        "wind_deg":    220,   # Karachi dominant: SW sea breeze
        "clouds":      20,
        "pressure":    avg_press,
    }


# ── Fetch real AQ history ──────────────────────────────────────────────────────

def fetch_aq_history(start_unix: int, end_unix: int) -> list:
    """
    Call OWM /air_pollution/history.
    Returns list of hourly records [{dt, components, main.aqi}, ...].
    Free tier supports this back to Nov 27 2020.
    """
    url = "http://api.openweathermap.org/data/2.5/air_pollution/history"
    resp = requests.get(url, params={
        "lat":   LAT,
        "lon":   LON,
        "start": start_unix,
        "end":   end_unix,
        "appid": OPENWEATHER_API_KEY,
    }, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    records = data.get("list", [])
    print(f"   ✔ Fetched {len(records)} hourly AQ records "
          f"({datetime.fromtimestamp(start_unix).date()} → "
          f"{datetime.fromtimestamp(end_unix).date()})")
    return records


def fetch_weather_history_onecall(unix_ts: int) -> dict | None:
    """
    Paid OWM OnecCall 3.0 timemachine endpoint.
    Only called if USE_PAID_WEATHER=True.
    """
    url = "https://api.openweathermap.org/data/3.0/onecall/timemachine"
    try:
        resp = requests.get(url, params={
            "lat":   LAT,
            "lon":   LON,
            "dt":    unix_ts,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric",
        }, timeout=15)
        resp.raise_for_status()
        d = resp.json()
        # The response has a 'data' array with one hourly entry
        entry = d.get("data", [{}])[0]
        return {
            "temperature": entry.get("temp", None),
            "feels_like":  entry.get("feels_like", None),
            "temp_min":    entry.get("temp", None),   # not available in timemachine
            "temp_max":    entry.get("temp", None),
            "temp_range":  0.0,
            "humidity":    entry.get("humidity", None),
            "wind_speed":  entry.get("wind_speed", None),
            "wind_deg":    entry.get("wind_deg", 0),
            "clouds":      entry.get("clouds", 0),
            "pressure":    entry.get("pressure", None),
        }
    except Exception as e:
        print(f"   ⚠️  Paid weather fetch failed for ts={unix_ts}: {e}")
        return None


# ── Build feature rows ─────────────────────────────────────────────────────────

def build_features_from_aq_records(aq_records: list) -> pd.DataFrame:
    """
    Convert raw OWM AQ history records into the feature DataFrame
    used for model training.
    """
    rows = []
    last_pm25 = None

    # Cache paid weather calls to avoid hammering the API (1 call per day max)
    weather_cache = {}   # date → weather dict

    for rec in aq_records:
        ts        = datetime.fromtimestamp(rec["dt"], tz=timezone.utc).replace(tzinfo=None)
        comp      = rec["components"]
        owm_aqi   = float(rec["main"]["aqi"])   # 1-5 OWM index, kept for reference

        pm2_5 = comp.get("pm2_5", 0.0)
        pm10  = comp.get("pm10",  0.0)
        no2   = comp.get("no2",   0.0)
        o3    = comp.get("o3",    0.0)
        so2   = comp.get("so2",   0.0)
        co    = comp.get("co",    0.0)

        # ── Time features ────────────────────────────────────────────────────
        hour        = ts.hour
        month       = ts.month
        day_of_week = ts.weekday()

        # ── Weather features ─────────────────────────────────────────────────
        if USE_PAID_WEATHER:
            date_key = ts.date()
            if date_key not in weather_cache:
                wx = fetch_weather_history_onecall(rec["dt"])
                weather_cache[date_key] = wx
                time.sleep(0.2)   # be kind to the API
            wx = weather_cache[date_key] or karachi_seasonal_temp(month, hour)
        else:
            # Free tier: use climatological normals — real shape, right scale
            wx = karachi_seasonal_temp(month, hour)

        # ── Derived features ─────────────────────────────────────────────────
        pm_ratio        = pm2_5 / pm10 if pm10 > 0 else 0.0
        temp_range      = wx["temp_range"] if "temp_range" in wx else (wx["temp_max"] - wx["temp_min"])
        aqi_change_rate = (pm2_5 - last_pm25) if last_pm25 is not None else 0.0
        last_pm25       = pm2_5

        # Convert PM2.5 to US AQI — this is our training TARGET
        us_aqi = pm25_to_us_aqi(pm2_5)

        rows.append({
            "timestamp":        ts,
            "hour":             hour,
            "day":              ts.day,
            "month":            month,
            "day_of_week":      day_of_week,
            "is_weekend":       int(day_of_week >= 5),
            "season":           (month % 12) // 3,
            "is_rush_hour":     int(hour in [7, 8, 9, 17, 18, 19]),
            "hour_sin":         round(np.sin(2 * np.pi * hour / 24), 6),
            "hour_cos":         round(np.cos(2 * np.pi * hour / 24), 6),
            # Pollutants (REAL from API)
            "pm2_5":            pm2_5,
            "pm10":             pm10,
            "no2":              no2,
            "o3":               o3,
            "so2":              so2,
            "co":               co,
            # Weather
            "temperature":      wx["temperature"],
            "feels_like":       wx["feels_like"],
            "temp_min":         wx["temp_min"],
            "temp_max":         wx["temp_max"],
            "temp_range":       temp_range,
            "pressure":         wx["pressure"],
            "humidity":         wx["humidity"],
            "wind_speed":       wx["wind_speed"],
            "wind_deg":         wx["wind_deg"],
            "clouds":           wx["clouds"],
            # Derived
            "pm_ratio":         round(pm_ratio, 6),
            "aqi_change_rate":  round(aqi_change_rate, 4),
            # Targets / reference
            "aqi":              owm_aqi,        # OWM 1-5 index (kept for reference)
        })

    df = pd.DataFrame(rows)
    return df


# ── Hopsworks upload ───────────────────────────────────────────────────────────

def push_to_hopsworks(df: pd.DataFrame):
    import hopsworks

    print(f"\n📤 Connecting to Hopsworks project: {HOPSWORKS_PROJECT}")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT)
    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description=f"Real AQI & weather features for {CITY} — backfilled from OWM history",
        primary_key=["timestamp"],
        event_time="timestamp",
    )

    print(f"📦 Inserting {len(df)} rows into '{FEATURE_GROUP_NAME}' v{FEATURE_GROUP_VERSION}...")
    fg.insert(df, write_options={"wait_for_job": False})
    print(f"✅ Done! {len(df)} records saved to Hopsworks.")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_backfill(days: int = DAYS_TO_FETCH):
    print("=" * 70)
    print(f"🚀 {CITY.upper()} AQI BACKFILL PIPELINE")
    print(f"   Fetching REAL historical data for the past {days} days")
    print(f"   Weather source: {'OWM OnceCall (paid)' if USE_PAID_WEATHER else 'Karachi climatological normals (free)'}")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # OWM history API accepts max ~1 year in one call, but we chunk by 30 days
    # to stay well within limits and provide progress feedback.
    now_ts    = int(datetime.now(timezone.utc).timestamp())
    start_ts  = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())

    all_records = []
    CHUNK_DAYS  = 30

    chunk_start = start_ts
    while chunk_start < now_ts:
        chunk_end = min(chunk_start + CHUNK_DAYS * 86400, now_ts)
        print(f"\n📡 Fetching chunk: "
              f"{datetime.fromtimestamp(chunk_start).strftime('%Y-%m-%d')} → "
              f"{datetime.fromtimestamp(chunk_end).strftime('%Y-%m-%d')}")
        try:
            records = fetch_aq_history(chunk_start, chunk_end)
            all_records.extend(records)
            time.sleep(0.5)   # gentle on the API
        except Exception as e:
            print(f"   ⚠️  Chunk failed: {e}. Skipping.")
        chunk_start = chunk_end

    if not all_records:
        print("❌ No records fetched. Check your API key and connectivity.")
        sys.exit(1)

    print(f"\n📊 Total raw records: {len(all_records)}")

    # Sort by time and deduplicate (OWM occasionally returns duplicates at chunk boundaries)
    all_records.sort(key=lambda r: r["dt"])
    seen = set()
    deduped = []
    for r in all_records:
        if r["dt"] not in seen:
            seen.add(r["dt"])
            deduped.append(r)
    print(f"📊 After deduplication: {len(deduped)} records")

    print(f"\n🔧 Building feature rows...")
    df = build_features_from_aq_records(deduped)

    print(f"\n📈 Feature stats:")
    print(f"   PM2.5 — min: {df['pm2_5'].min():.1f}  max: {df['pm2_5'].max():.1f}  "
          f"mean: {df['pm2_5'].mean():.1f}  std: {df['pm2_5'].std():.1f}")
    print(f"   Temp  — min: {df['temperature'].min():.1f}  max: {df['temperature'].max():.1f}")
    print(f"   Rows  — {len(df)}  |  Date range: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")

    if df['pm2_5'].std() < 1.0:
        print("\n⚠️  WARNING: PM2.5 std is very low — model will learn a constant function!")
        print("   This may mean all fetched values are the same. Check your API key.")

    print(f"\n💾 Uploading to Hopsworks...")
    push_to_hopsworks(df)

    print("\n✅ Backfill complete! Now run:")
    print("   python training_pipeline.py")
    print("=" * 70)


if __name__ == "__main__":
    if not OPENWEATHER_API_KEY:
        sys.exit("❌ OPENWEATHERMAP_API_KEY not found in .env")
    if not HOPSWORKS_API_KEY or not HOPSWORKS_PROJECT:
        sys.exit("❌ HOPSWORKS_API_KEY or HOPSWORKS_PROJECT_NAME not found in .env")

    run_backfill()
