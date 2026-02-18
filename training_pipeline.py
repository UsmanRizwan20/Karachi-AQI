import os
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from dotenv import load_dotenv

load_dotenv()

HOPSWORKS_API_KEY     = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT     = os.getenv("HOPSWORKS_PROJECT_NAME")
FEATURE_GROUP_NAME    = os.getenv("FEATURE_GROUP_NAME", "karachi_aqi_features")
FEATURE_GROUP_VERSION = int(os.getenv("FEATURE_GROUP_VERSION", "1"))

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


TARGET_COL   = 'pm2_5'
EXCLUDE_COLS = ['timestamp', 'aqi', 'pm2_5']   


def pm25_to_us_aqi(pm25: float) -> float:
    """
    Convert PM2.5 concentration (μg/m³) to US AQI using EPA linear interpolation.
    https://www.airnow.gov/sites/default/files/2020-05/aqi-technical-assistance-document-sept2018.pdf
    """
    breakpoints = [
        # (pm25_low, pm25_high, aqi_low, aqi_high)
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


def load_data_from_hopsworks() -> pd.DataFrame:
    print("\n" + "="*70)
    print("LOADING DATA FROM HOPSWORKS")
    print("="*70)
    try:
        import hopsworks
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY,
                                  project=HOPSWORKS_PROJECT)
        fs = project.get_feature_store()
        fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
        df = fg.read()
        print(f"✅ Loaded {len(df)} records from Hopsworks")
        return df
    except Exception as e:
        print(f"❌ Hopsworks error: {e}")
        sys.exit(1)


def prepare_data(df):
    # Sanity-check target variance — this was the root cause of the original bug
    target_std = df[TARGET_COL].std()
    target_min = df[TARGET_COL].min()
    target_max = df[TARGET_COL].max()
    print(f"\n📊 Target '{TARGET_COL}' stats:")
    print(f"   min={target_min:.2f}  max={target_max:.2f}  std={target_std:.2f}")
    if target_std < 0.5:
        print(f"⚠️  WARNING: Target has very low variance (std={target_std:.4f}).")
        print("   The model will learn a constant function and all forecasts will be identical.")
        print("   Check that your feature pipeline is collecting diverse data over time.")

    df = df.dropna(subset=[TARGET_COL])
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feature_cols].fillna(df[feature_cols].mean())
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save the exact feature list so app.py uses the same columns in the same order
    with open(MODEL_DIR / "feature_names.json", "w") as f:
        json.dump(feature_cols, f)
    print(f"\n💾 Saved feature_names.json with {len(feature_cols)} features: {feature_cols}")

    return X_train, X_test, y_train, y_test, feature_cols


def train_random_forest(X_train, y_train, X_test, y_test, feature_names):
    print("\n" + "="*70)
    print("🌲 MODEL 1: RANDOM FOREST")
    print("="*70)
    model = RandomForestRegressor(n_estimators=100, max_depth=20,
                                  min_samples_split=5, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    print(f"   RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")

    fi = pd.DataFrame({'feature': feature_names,
                       'importance': model.feature_importances_}).sort_values('importance', ascending=False)
    print("🔝 Top 5 Features:")
    for _, row in fi.head().iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")

    # Spot-check: predictions should NOT all be the same value
    sample_preds = model.predict(X_test.iloc[:10])
    print(f"   Sample predictions (pm2_5): {[round(p,1) for p in sample_preds]}")

    joblib.dump(model, MODEL_DIR / "random_forest_model.pkl")
    print("💾 Saved: models/random_forest_model.pkl")
    return model, {'model': 'Random Forest', 'rmse': rmse, 'mae': mae, 'r2': r2}


def train_xgboost(X_train, y_train, X_test, y_test, feature_names):
    print("\n" + "="*70)
    print("🚀 MODEL 2: XGBOOST")
    print("="*70)
    model = xgb.XGBRegressor(n_estimators=100, max_depth=10,
                             learning_rate=0.1, subsample=0.8,
                             colsample_bytree=0.8, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    print(f"   RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")
    sample_preds = model.predict(X_test.iloc[:10])
    print(f"   Sample predictions (pm2_5): {[round(p,1) for p in sample_preds]}")
    joblib.dump(model, MODEL_DIR / "xgboost_model.pkl")
    print("💾 Saved: models/xgboost_model.pkl")
    return model, {'model': 'XGBoost', 'rmse': rmse, 'mae': mae, 'r2': r2}


def train_neural_network(X_train, y_train, X_test, y_test, feature_names):
    print("\n" + "="*70)
    print("🧠 MODEL 3: NEURAL NETWORK (MLPRegressor)")
    print("="*70)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    model = MLPRegressor(hidden_layer_sizes=(64, 32, 16),
                         activation='relu', solver='adam',
                         max_iter=500, random_state=42,
                         early_stopping=True, validation_fraction=0.2)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    print(f"   RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")
    print(f"   Stopped at epoch: {model.n_iter_}")
    sample_preds = model.predict(X_test_scaled[:10])
    print(f"   Sample predictions (pm2_5): {[round(p,1) for p in sample_preds]}")
    joblib.dump(model,  MODEL_DIR / "neural_network_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "neural_network_scaler.pkl")
    print("💾 Saved: models/neural_network_model.pkl + scaler")
    return model, {'model': 'Neural Network (MLP)', 'rmse': rmse, 'mae': mae, 'r2': r2}


def run_training_pipeline():
    print("\n" + "="*70)
    print("🚀 KARACHI AQI TRAINING PIPELINE")
    print(f"   Target variable: '{TARGET_COL}' (PM2.5 μg/m³ → converted to US AQI in app)")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    df = load_data_from_hopsworks()
    if len(df) < 10:
        print("❌ Not enough data!")
        return

    X_train, X_test, y_train, y_test, feature_names = prepare_data(df)

    all_metrics = []
    _, m1 = train_random_forest(X_train, y_train, X_test, y_test, feature_names)
    all_metrics.append(m1)
    _, m2 = train_xgboost(X_train, y_train, X_test, y_test, feature_names)
    all_metrics.append(m2)
    _, m3 = train_neural_network(X_train, y_train, X_test, y_test, feature_names)
    all_metrics.append(m3)

    print("\n" + "="*70)
    print("📊 FINAL MODEL COMPARISON")
    print("="*70)
    metrics_df = pd.DataFrame(all_metrics)
    print(metrics_df.to_string(index=False))
    best = metrics_df.loc[metrics_df['rmse'].idxmin()]
    print(f"\n🏆 Best Model : {best['model']}")
    print(f"   RMSE: {best['rmse']:.4f} | MAE: {best['mae']:.4f} | R²: {best['r2']:.4f}")

    with open(MODEL_DIR / "best_model.txt", "w") as f:
        f.write(best['model'])
    print(f"\n💾 Saved best_model.txt: {best['model']}")
    print("\n✅ ALL 3 MODELS TRAINED AND SAVED!")
    print("Now run: streamlit run app.py")
    print("="*70)


if __name__ == "__main__":
    run_training_pipeline()
