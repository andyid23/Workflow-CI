import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, json

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

# ====== SETUP LOCAL MLFLOW TRACKING ======
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Workflow-CI_RF_Tuning")

# ====== LOAD DATA ======
DATA_PATH = "namadataset_preprocessing/processed_data.csv"
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["Post_Semester_GPA"])
y = df["Post_Semester_GPA"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ====== HYPERPARAMETER TUNING ======
param_grid = {
    "n_estimators": [100, 150],
    "max_depth": [10, 15],
    "min_samples_split": [2, 5]
}

rf = RandomForestRegressor(random_state=42)
grid = GridSearchCV(rf, param_grid, cv=3, scoring="neg_mean_squared_error", n_jobs=-1)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

# ====== EVALUASI ======
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Best params: {grid.best_params_}")
print(f"MSE: {mse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")

# ====== SAVE MODEL & METRICS KE REPO (bukan ke server) ======
os.makedirs("model", exist_ok=True)
model_path = "model/rf_model.joblib"
joblib.dump(best_model, model_path, compress=3)

# Simpan metrics sebagai JSON
metrics = {
    "best_params": {k: v for k, v in grid.best_params_.items()},
    "mse": mse,
    "mae": mae,
    "r2": r2
}
with open("model/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# === FEATURE IMPORTANCE PLOT ===
importances = best_model.feature_importances_
feats = X.columns
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(feats, importances, color="steelblue")
ax.set_xlabel("Feature Importance")
ax.set_title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig("model/feature_importance.png", dpi=150)
plt.close()

# === ACTUAL vs PREDICTED PLOT ===
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_test, y_pred, alpha=0.6, edgecolors="k", linewidth=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
ax.set_xlabel("Actual Post_Semester_GPA")
ax.set_ylabel("Predicted Post_Semester_GPA")
ax.set_title("Actual vs Predicted")
plt.tight_layout()
plt.savefig("model/actual_vs_predicted.png", dpi=150)
plt.close()

# ====== LOG KE MLFLOW (local file) ======
with mlflow.start_run(run_name="Workflow-CI_Tuned"):
    for k, v in grid.best_params_.items():
        mlflow.log_param(k, v)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    mlflow.log_artifact(model_path)
    mlflow.log_artifact("model/metrics.json")
    mlflow.log_artifact("model/feature_importance.png")
    mlflow.log_artifact("model/actual_vs_predicted.png")
    mlflow.log_artifact(DATA_PATH)

    print(f"✅ Run selesai — Artifacts disimpan di folder model/ dan mlruns/")
    print(f"📌 Lihat hasil: ls model/")

print(f"✅ Run selesai — Artifacts disimpan di folder model/ dan mlruns/")
print(f"📌 Lihat hasil: ls model/")
