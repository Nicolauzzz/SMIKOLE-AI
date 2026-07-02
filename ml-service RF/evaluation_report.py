"""
Comprehensive evaluation script for Random Forest model.
Generates all data needed for the academic report tables.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
import os

# ============================================================
# 1. Load & Split Data (same as training.py)
# ============================================================
data = pd.read_csv("dataset.csv")
features = ["DOC", "populasi", "bobot_awal_per_ekor_gr", "pakan_harian_gr", "panjang_periode_hari"]
X = data[features]
y = data["delta_bobot_per_ekor_gr"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ============================================================
# 2. Train Model (reproduce exact same model)
# ============================================================
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ============================================================
# 3. Predictions
# ============================================================
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# ============================================================
# 4. TABEL: Rincian Dataset
# ============================================================
print("=" * 70)
print("TABEL: Rincian Uji Dataset")
print("=" * 70)
print(f"{'Parameter':<35} {'Nilai'}")
print("-" * 70)
print(f"{'Total data':<35} {len(data)} data")
print(f"{'Fitur Input':<35} {', '.join(features)}")
print(f"{'Target Output':<35} delta_bobot_per_ekor_gr")
print(f"{'Rentang DOC':<35} {data['DOC'].min()} - {data['DOC'].max()} hari")
print(f"{'Rentang Populasi':<35} {data['populasi'].min()} - {data['populasi'].max()} ekor")
print(f"{'Rentang Bobot Awal':<35} {data['bobot_awal_per_ekor_gr'].min():.2f} - {data['bobot_awal_per_ekor_gr'].max():.2f} gr/ekor")
print(f"{'Rentang Pakan Harian':<35} {data['pakan_harian_gr'].min()} - {data['pakan_harian_gr'].max()} gr")
print(f"{'Rentang Delta Bobot':<35} {data['delta_bobot_per_ekor_gr'].min():.2f} - {data['delta_bobot_per_ekor_gr'].max():.2f} gr/ekor")
print(f"{'Data Training':<35} {len(X_train)} sampel (80%)")
print(f"{'Data Testing':<35} {len(X_test)} sampel (20%)")

# ============================================================
# 5. TABEL: Hasil Evaluasi Model
# ============================================================
mae_train = mean_absolute_error(y_train, y_pred_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
r2_train = r2_score(y_train, y_pred_train)

mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_test = r2_score(y_test, y_pred_test)

print(f"\n{'=' * 70}")
print("TABEL: Hasil Evaluasi Model Random Forest")
print("=" * 70)
print(f"{'Metrik':<25} {'Training':<20} {'Testing'}")
print("-" * 70)
print(f"{'MAE (gram/ekor)':<25} {mae_train:.4f}{'':<14} {mae_test:.4f}")
print(f"{'RMSE (gram/ekor)':<25} {rmse_train:.4f}{'':<14} {rmse_test:.4f}")
print(f"{'R² Score':<25} {r2_train:.4f}{'':<14} {r2_test:.4f}")

# Error relatif terhadap rentang data
range_delta = data['delta_bobot_per_ekor_gr'].max() - data['delta_bobot_per_ekor_gr'].min()
error_relatif = (mae_test / range_delta) * 100
print(f"\n{'Rentang Delta Bobot':<25} {range_delta:.2f} gram/ekor")
print(f"{'Error Relatif (MAE/Range)':<25} {error_relatif:.2f}%")

# ============================================================
# 6. TABEL: Feature Importance
# ============================================================
print(f"\n{'=' * 70}")
print("TABEL: Feature Importance")
print("=" * 70)
print(f"{'Peringkat':<10} {'Fitur':<30} {'Importance':<15} {'Persentase'}")
print("-" * 70)
importances = sorted(zip(features, model.feature_importances_), key=lambda x: -x[1])
for i, (feat, imp) in enumerate(importances, 1):
    print(f"{i:<10} {feat:<30} {imp:.4f}{'':<9} {imp*100:.2f}%")

# ============================================================
# 7. TABEL: 10 Sampel Prediksi vs Aktual
# ============================================================
print(f"\n{'=' * 70}")
print("TABEL: 10 Sampel Prediksi vs Aktual (Data Test)")
print("=" * 70)

test_results = X_test.copy()
test_results['aktual'] = y_test.values
test_results['prediksi'] = y_pred_test
test_results['error'] = abs(y_test.values - y_pred_test)
test_results = test_results.reset_index(drop=True)

print(f"{'No':<4} {'DOC':<6} {'Pop':<6} {'Bobot(gr)':<12} {'Pakan(gr)':<12} {'Periode':<9} {'Prediksi':<12} {'Aktual':<12} {'Error'}")
print("-" * 90)
for i in range(min(10, len(test_results))):
    row = test_results.iloc[i]
    print(f"{i+1:<4} {int(row['DOC']):<6} {int(row['populasi']):<6} {row['bobot_awal_per_ekor_gr']:<12.2f} {int(row['pakan_harian_gr']):<12} {int(row['panjang_periode_hari']):<9} {row['prediksi']:<12.2f} {row['aktual']:<12.2f} {row['error']:.2f}")

# Statistik error
print(f"\n--- Statistik Error pada 10 Sampel ---")
sample_errors = test_results['error'].head(10)
print(f"Error Minimum : {sample_errors.min():.2f} gram")
print(f"Error Maksimum: {sample_errors.max():.2f} gram")
print(f"Error Rata-rata: {sample_errors.mean():.2f} gram")

# ============================================================
# 8. TABEL: Semua Data Test (untuk kelengkapan)
# ============================================================
print(f"\n{'=' * 70}")
print(f"Statistik Lengkap Data Test ({len(test_results)} sampel)")
print("=" * 70)
print(f"Error < 0.5 gram : {(test_results['error'] < 0.5).sum()} sampel ({(test_results['error'] < 0.5).sum()/len(test_results)*100:.1f}%)")
print(f"Error < 1.0 gram : {(test_results['error'] < 1.0).sum()} sampel ({(test_results['error'] < 1.0).sum()/len(test_results)*100:.1f}%)")
print(f"Error < 2.0 gram : {(test_results['error'] < 2.0).sum()} sampel ({(test_results['error'] < 2.0).sum()/len(test_results)*100:.1f}%)")
print(f"Error >= 2.0 gram: {(test_results['error'] >= 2.0).sum()} sampel ({(test_results['error'] >= 2.0).sum()/len(test_results)*100:.1f}%)")

# ============================================================
# 9. Cross-validation (bonus untuk laporan)
# ============================================================
from sklearn.model_selection import cross_val_score

print(f"\n{'=' * 70}")
print("TABEL: Cross-Validation (5-Fold)")
print("=" * 70)

cv_mae = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
cv_rmse = np.sqrt(-cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error'))
cv_r2 = cross_val_score(model, X, y, cv=5, scoring='r2')

print(f"{'Fold':<8} {'MAE':<15} {'RMSE':<15} {'R²'}")
print("-" * 50)
for i in range(5):
    print(f"Fold {i+1:<3} {cv_mae[i]:<15.4f} {cv_rmse[i]:<15.4f} {cv_r2[i]:.4f}")
print("-" * 50)
print(f"{'Mean':<8} {cv_mae.mean():<15.4f} {cv_rmse.mean():<15.4f} {cv_r2.mean():.4f}")
print(f"{'Std':<8} {cv_mae.std():<15.4f} {cv_rmse.std():<15.4f} {cv_r2.std():.4f}")

# ============================================================
# 10. Quick Prediction Test Scenarios (for API testing)
# ============================================================
print(f"\n{'=' * 70}")
print("Pengujian Skenario Prediksi (Preview API)")
print("=" * 70)

test_scenarios = [
    {"name": "Kondisi Normal", "DOC": 30, "populasi": 2000, "bobot_awal_per_ekor_gr": 15.0, "pakan_harian_gr": 1500, "panjang_periode_hari": 10},
    {"name": "Fase Awal", "DOC": 5, "populasi": 3000, "bobot_awal_per_ekor_gr": 0.5, "pakan_harian_gr": 100, "panjang_periode_hari": 10},
    {"name": "Golden Age", "DOC": 10, "populasi": 2500, "bobot_awal_per_ekor_gr": 2.0, "pakan_harian_gr": 300, "panjang_periode_hari": 10},
    {"name": "Fase Akhir", "DOC": 80, "populasi": 2000, "bobot_awal_per_ekor_gr": 50.0, "pakan_harian_gr": 5000, "panjang_periode_hari": 10},
    {"name": "Overfeeding", "DOC": 30, "populasi": 2000, "bobot_awal_per_ekor_gr": 5.0, "pakan_harian_gr": 5000, "panjang_periode_hari": 10},
]

for sc in test_scenarios:
    name = sc.pop("name")
    inp = pd.DataFrame([sc])
    pred = model.predict(inp)[0]
    
    # Calculate FCR
    delta_biomassa = sc['populasi'] * pred
    total_pakan = sc['pakan_harian_gr'] * sc['panjang_periode_hari']
    fcr = total_pakan / delta_biomassa if delta_biomassa > 0 else float('inf')
    
    if fcr < 1.0:
        status = "Efisien"
    elif fcr <= 1.3:
        status = "Normal"
    else:
        status = "Overfeeding"
    
    adg = pred / sc['panjang_periode_hari']
    
    print(f"\n--- {name} ---")
    print(f"  Input: DOC={sc['DOC']}, pop={sc['populasi']}, bobot={sc['bobot_awal_per_ekor_gr']}g, pakan={sc['pakan_harian_gr']}g, hari={sc['panjang_periode_hari']}")
    print(f"  Pred delta bobot: {pred:.2f} gr/ekor")
    print(f"  ADG: {adg:.2f} gr/ekor/hari")
    print(f"  FCR: {fcr:.2f}")
    print(f"  Status: {status}")

print(f"\n{'=' * 70}")
print("SELESAI - Semua data untuk tabel laporan telah dihasilkan")
print("=" * 70)
