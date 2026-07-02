import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import matplotlib.pyplot as plt

# ========================================
# KONFIGURASI
# ========================================
TIME_STEPS = 30  
N_FEATURES = 3   
EPOCHS = 100
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

# ========================================
# 1. LOAD DATA
# ========================================
print("📂 Loading data...")
# Ganti dengan path file CSV kamu
# Format CSV: timestamp, temperature, pH, DO
# Pastikan data TIDAK termasuk periode saat heater menyala
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "water_quality_data.csv")
df = pd.read_csv(csv_path)

# Ambil kolom yang dibutuhkan
data = df[['temperature', 'pH', 'DO']].values
print(f"✅ Data shape: {data.shape}")
print(f"   Temperature range: {data[:, 0].min():.2f} - {data[:, 0].max():.2f}°C")
print(f"   pH range: {data[:, 1].min():.2f} - {data[:, 1].max():.2f}")
print(f"   DO range: {data[:, 2].min():.2f} - {data[:, 2].max():.2f}")

# ========================================
# 2. NORMALISASI DATA
# ========================================
print("\n🔧 Normalizing data...")
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# ========================================
# 3. CREATE SEQUENCES
# ========================================
def create_sequences(data, time_steps):
    """
    Membuat sequence untuk LSTM
    Input: 30 timesteps → Output: timestep ke-31
    """
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

print(f"\n📊 Creating sequences with TIME_STEPS={TIME_STEPS}...")
X, y = create_sequences(data_scaled, TIME_STEPS)
print(f"✅ X shape: {X.shape}")  # (samples, 30, 3)
print(f"✅ y shape: {y.shape}")  # (samples, 3)

# ========================================
# 4. SPLIT DATA
# ========================================
print("\n✂️ Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False  # shuffle=False untuk time series
)
print(f"✅ Training samples: {len(X_train)}")
print(f"✅ Testing samples: {len(X_test)}")

# ========================================
# 5. BUILD MODEL
# ========================================
print("\n🏗️ Building LSTM model...")
model = Sequential([
    # Layer 1: LSTM dengan 64 units
    LSTM(64, return_sequences=True, input_shape=(TIME_STEPS, N_FEATURES)),
    Dropout(0.2),
    
    # Layer 2: LSTM dengan 32 units
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    
    # Layer 3: Dense layer
    Dense(16, activation='relu'),
    
    # Output layer: 3 features (temperature, pH, DO)
    Dense(N_FEATURES)
])

model.compile(
    optimizer='adam',
    loss='mse',  # Mean Squared Error
    metrics=['mae']  # Mean Absolute Error
)

model.summary()

# ========================================
# 6. CALLBACKS
# ========================================
callbacks = [
    # Stop training jika tidak ada improvement
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),
    
    # Save model terbaik
    ModelCheckpoint(
        'lstm_water_quality_best.h5',
        monitor='val_loss',
        save_best_only=True
    )
]

# ========================================
# 7. TRAINING
# ========================================
print("\n🚀 Training model...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    callbacks=callbacks,
    verbose=1
)

# ========================================
# 8. EVALUASI
# ========================================
print("\n📈 Evaluating model...")
train_loss, train_mae = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)

print(f"\n✅ Training Loss: {train_loss:.4f}, MAE: {train_mae:.4f}")
print(f"✅ Testing Loss: {test_loss:.4f}, MAE: {test_mae:.4f}")

# ========================================
# 9. TEST PREDICTION
# ========================================
print("\n🔮 Testing prediction...")
sample_prediction_scaled = model.predict(X_test[:5])
sample_prediction = scaler.inverse_transform(sample_prediction_scaled)
sample_actual = scaler.inverse_transform(y_test[:5])

print("\nSample Predictions vs Actual:")
for i in range(5):
    print(f"\nSample {i+1}:")
    print(f"  Predicted - Temp: {sample_prediction[i][0]:.2f}°C, pH: {sample_prediction[i][1]:.2f}, DO: {sample_prediction[i][2]:.2f}")
    print(f"  Actual    - Temp: {sample_actual[i][0]:.2f}°C, pH: {sample_actual[i][1]:.2f}, DO: {sample_actual[i][2]:.2f}")
    print(f"  Error     - Temp: {abs(sample_prediction[i][0] - sample_actual[i][0]):.2f}°C")

# ========================================
# 10. SAVE MODEL & SCALER
# ========================================
print("\n💾 Saving model and scaler...")
model.save("lstm_water_quality.h5")
joblib.dump(scaler, "scaler_water_quality.pkl")
print("✅ Model saved: lstm_water_quality.h5")
print("✅ Scaler saved: scaler_water_quality.pkl")

# ========================================
# 11. PLOT TRAINING HISTORY + TEST METRICS
# ========================================
print("\n📊 Plotting training history...")
from sklearn.metrics import mean_squared_error, mean_absolute_error

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Grafik 1: Loss (MSE) ---
epochs_range = range(1, len(history.history['loss']) + 1)
axes[0].plot(epochs_range, history.history['loss'], 'b-', linewidth=1.5, label='Training Loss')
axes[0].plot(epochs_range, history.history['val_loss'], 'orange', linewidth=1.5, label='Validation Loss')
axes[0].axhline(y=test_loss, color='red', linestyle='--', linewidth=1.5, label=f'Test Loss = {test_loss:.4f}')
axes[0].set_title('Model Loss (MSE) — Training vs Validation vs Testing', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (MSE)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# --- Grafik 2: MAE ---
axes[1].plot(epochs_range, history.history['mae'], 'b-', linewidth=1.5, label='Training MAE')
axes[1].plot(epochs_range, history.history['val_mae'], 'orange', linewidth=1.5, label='Validation MAE')
axes[1].axhline(y=test_mae, color='red', linestyle='--', linewidth=1.5, label=f'Test MAE = {test_mae:.4f}')
axes[1].set_title('Model MAE — Training vs Validation vs Testing', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MAE')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Training history saved: training_history.png")

# ========================================
# 12. PLOT PREDICTION vs ACTUAL
# ========================================
print("📊 Plotting prediction vs actual...")

# Prediksi seluruh test set
all_pred_scaled = model.predict(X_test, verbose=0)
all_pred = scaler.inverse_transform(all_pred_scaled)
all_actual = scaler.inverse_transform(y_test)

param_names = ['Temperature (°C)', 'pH', 'DO (mg/L)']
param_units = ['°C', '', 'mg/L']
colors = ['#e74c3c', '#2ecc71', '#3498db']

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

for i, (name, unit, color) in enumerate(zip(param_names, param_units, colors)):
    mae_val = mean_absolute_error(all_actual[:, i], all_pred[:, i])
    rmse_val = np.sqrt(mean_squared_error(all_actual[:, i], all_pred[:, i]))
    
    axes[i].plot(all_actual[:, i], color='black', linewidth=0.8, alpha=0.7, label='Aktual')
    axes[i].plot(all_pred[:, i], color=color, linewidth=0.8, alpha=0.7, label='Prediksi')
    axes[i].set_title(f'{name} — Prediksi vs Aktual  |  MAE: {mae_val:.4f}  |  RMSE: {rmse_val:.4f}', 
                       fontsize=11, fontweight='bold')
    axes[i].set_xlabel('Sample Index')
    axes[i].set_ylabel(f'{name}')
    axes[i].legend(loc='upper right')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('prediction_vs_actual.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Prediction vs actual saved: prediction_vs_actual.png")

print("\n✨ Training complete!")
