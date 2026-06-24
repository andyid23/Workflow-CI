import sys, io, os
import joblib
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

"""
modelling.py — MLflow Autolog + Save model.joblib

Jika dijalankan via `mlflow run MLProject --env-manager=local`:
  - MLFLOW_RUN_ID sudah diset otomatis oleh MLflow CLI
  - Tidak perlu start_run() / set_experiment()
  - mlflow.autolog() hook ke sklearn.fit()

Jika dijalankan standalone:
  - Set tracking URI ke file:../mlruns
  - Buat experiment sendiri
"""

# Cek apakah dijalankan via MLflow CLI
VIA_MLFLOW_CLI = "MLFLOW_RUN_ID" in os.environ

if not VIA_MLFLOW_CLI:
    mlflow.set_tracking_uri("file:../mlruns")
    mlflow.set_experiment("Workflow-CI_RF_Autolog")

# ====== AKTIFKAN AUTOLOG ======
mlflow.autolog()

# ====== LOAD DATA ======
df = pd.read_csv('namadataset_preprocessing/processed_data.csv')
target_col = 'Skill_Retention_Score'
X = df.drop(columns=[target_col])
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ====== TRAINING ======
print(f"🔄 Training RandomForest dengan autolog...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f"✅ Selesai! MSE={mse:.4f} R2={r2:.4f}")

# ====== SAVE MODEL UNTUK DOCKER BUILD ======
os.makedirs("../model", exist_ok=True)
joblib.dump(model, "../model/rf_model.joblib", compress=3)
print(f"✅ Model disimpan ke ../model/rf_model.joblib")
print(f"📌 Cek hasil di mlflow UI (mlruns/)")
