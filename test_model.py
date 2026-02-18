import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

m = joblib.load('models/random_forest_model.pkl')

for i in range(1, 4):
    future = datetime.now() + timedelta(days=i)
    row = {
        'hour': future.hour,
        'day': future.day,
        'month': future.month,
        'day_of_week': future.weekday(),
        'is_weekend': int(future.weekday() >= 5),
        'season': (future.month % 12) // 3,
        'pm2_5': 50,
        'pm10': 80,
        'no2': 10,
        'o3': 20,
        'so2': 5,
        'co': 200,
        'temperature': 28,
        'feels_like': 26,
        'temp_min': 25,
        'temp_max': 31,
        'pressure': 1010,
        'humidity': 60,
        'wind_speed': 3,
        'wind_deg': 0,
        'clouds': 0,
        'pm_ratio': 0.625,
        'temp_range': 6,
        'is_rush_hour': 0,
        'hour_sin': np.sin(2 * np.pi * future.hour / 24),
        'hour_cos': np.cos(2 * np.pi * future.hour / 24),
    }
    df = pd.DataFrame([row])
    print(f"Day {i} ({future.strftime('%A')}): AQI = {m.predict(df)[0]:.4f}")
