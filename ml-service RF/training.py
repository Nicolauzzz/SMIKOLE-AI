import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load dataset
data = pd.read_csv("dataset.csv")
print(f"Dataset: {len(data)} records")
print(f"Columns: {list(data.columns)}\n")

# Features and target
features = ["DOC", "populasi", "bobot_awal_per_ekor_gr", "pakan_harian_gr", "panjang_periode_hari"]
X = data[features]
y = data["delta_bobot_per_ekor_gr"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# Train model
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n=== Model Evaluation ===")
print(f"MAE:  {mae:.2f} gram/ekor")
print(f"RMSE: {rmse:.2f} gram/ekor")
print(f"R²:   {r2:.4f}")

# Feature importance
print(f"\n=== Feature Importance ===")
for feat, imp in sorted(zip(features, model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {feat}: {imp:.4f}")

# Save model
model_path = "model_growth.pkl"
joblib.dump(model, model_path)
print(f"\nModel saved to {model_path}")

# Quick test prediction
test_input = pd.DataFrame([{
    "DOC": 30,
    "populasi": 2000,
    "bobot_awal_per_ekor_gr": 15.0,
    "pakan_harian_gr": 1500,
    "panjang_periode_hari": 10
}])
pred = model.predict(test_input)[0]
print(f"\n=== Test Prediction ===")
print(f"Input: DOC=30, populasi=2000, bobot/ekor=15.0g, pakan/hari=1500g, hari=10")
print(f"Predicted delta_bobot_per_ekor: {pred:.2f} gram")