from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import joblib
import os
import firebase_admin
from firebase_admin import credentials, firestore as admin_firestore
from datetime import datetime
from google.cloud import firestore as g_firestore
import time

app = Flask(__name__)

API_KEY = os.environ.get("API_KEY")

if not firebase_admin._apps:
    cred = credentials.ApplicationDefault()
    firebase_admin.initialize_app(cred)

db = admin_firestore.client()

@app.before_request
def check_api_key():
    if request.path == "/health":
        return
    key = request.headers.get("x-api-key")
    if key != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

# Load model dan scaler
model = load_model("lstm_water_quality.h5", compile=False)
scaler = joblib.load("scaler_water_quality.pkl")

TIME_STEPS = 30
N_FEATURES = 3  # temperature, pH, DO

# ========================================
# PARAMETER THRESHOLDS
# ========================================
TEMP_MIN = 25.0  # °C
TEMP_MAX = 30.0  # °C
PH_MIN = 7.0
PH_MAX = 8.5
DO_MIN = 3.0  # mg/L

@app.route("/health", methods=["GET"])
def health():
    if model is None or scaler is None:
        return "MODEL NOT READY", 500
    return "OK", 200

def run_ai(sequence, pond_id=None):

    latest_temp = float(sequence[-1][0])
    latest_ph = float(sequence[-1][1])
    latest_do = float(sequence[-1][2])

    sequence_scaled = scaler.transform(sequence)
    sequence_scaled = sequence_scaled.reshape(1, TIME_STEPS, N_FEATURES)

    prediction_scaled = model.predict(sequence_scaled, verbose=0)
    prediction = scaler.inverse_transform(prediction_scaled)

    predicted_temp = float(prediction[0][0])
    predicted_ph = float(prediction[0][1])
    predicted_do = float(prediction[0][2])
    
    if latest_temp < TEMP_MIN or predicted_temp < TEMP_MIN:
            heater_status = "ON"
            temp_status = "low"
    elif latest_temp > TEMP_MAX:
            heater_status = "OFF"
            temp_status = "high"
    else:
            heater_status = "OFF"
            temp_status = "normal"

    # 🔹 VALIDASI pH
    if latest_ph < PH_MIN or latest_ph > PH_MAX:
        ph_warning = "pH tidak normal"
    else:
            ph_warning = None

        # 🔹 VALIDASI DO
    if latest_do < DO_MIN:
            do_warning = "DO rendah"
    else:
            do_warning = None

        # 🔹 STATUS RISIKO KESELURUHAN
    warnings = []
    if temp_status != "normal":
            warnings.append(f"temperature_{temp_status}")
    if ph_warning:
            warnings.append("ph_abnormal")
    if do_warning:
            warnings.append("do_low")

    if warnings:
            risk_status = "warning"
    else:
            risk_status = "normal"

    # 🔹 SISTEM REKOMENDASI OTOMATIS
    recommendations = []
        
        # Rekomendasi berdasarkan suhu
    if temp_status == "low":
            if predicted_temp < latest_temp:
                recommendations.append("⚠️ Suhu diprediksi akan menurun dalam 30 menit kedepan, Pastikan heater berfungsi dengan baik.")
            else:
                recommendations.append("✅ Heater aktif, suhu akan naik dalam beberapa saat.")
    elif temp_status == "high":
            recommendations.append("🔥 Suhu terlalu tinggi! Matikan heater dan periksa kondisi kolam.")
            if predicted_temp > latest_temp:
                recommendations.append("⚠️ Suhu diprediksi akan naik dalam 30 menit kedepan, Segera ambil tindakan.")
        
        # Rekomendasi berdasarkan pH
    if ph_warning:
            if latest_ph < PH_MIN:
                recommendations.append("⚖️ pH berada di bawah rentang aman. Kondisi ini dapat mempengaruhi keseimbangan fisiologis ikan.")
            elif latest_ph > PH_MAX:
                recommendations.append("🧪 pH berada di atas batas optimal. Lakukan penyesuaian bertahap untuk stabilisasi.")
        
        # Rekomendasi berdasarkan DO
    if do_warning:
            recommendations.append("💨 Kadar oksigen terlarut rendah. Tingkatkan aerasi atau aktifkan aerator tambahan.")
            if predicted_do < latest_do:
                recommendations.append("📉 Prediksi menunjukkan potensi penurunan oksigen lebih lanjut. Risiko stres meningkat.")
            else:
                recommendations.append("🔄 Sistem aerasi diperkirakan membantu menjaga kestabilan oksigen.")
        
        # Rekomendasi preventif
    if risk_status == "normal":
            # Cek apakah ada tren berbahaya meskipun saat ini normal
            temp_change = predicted_temp - latest_temp
            if abs(temp_change) > 2.0:
                if temp_change > 0:
                    recommendations.append("📊 Tren: Suhu akan naik +{:.1f}°C. Monitor terus.".format(temp_change))
                else:
                    recommendations.append("📊 Tren: Suhu akan turun {:.1f}°C. Monitor terus.".format(temp_change))
        
        # Jika semua normal dan stabil
    if not recommendations:
            recommendations.append("✅ Semua parameter normal dan stabil. Sistem berjalan baik.")

    result = {
            "predicted_temperature_30min": round(predicted_temp, 2),
            "predicted_ph_30min": round(predicted_ph, 2),
            "predicted_do_30min": round(predicted_do, 2),
            "heater_status": heater_status,
            "temperature_status": temp_status,
            "risk_status": risk_status,
            "recommendations": recommendations,
        }

    # ==========================================
    # AUTO CONTROL (FIXED & SAFE)
    # ==========================================
    if pond_id:

        control_ref = db.collection("ponds").document(pond_id)\
            .collection("control").document("settings")

        control_doc = control_ref.get()

        if control_doc.exists:

            control_data = control_doc.to_dict()
            mode = control_data.get("mode", "AUTO")

            if mode == "AUTO":

                manual_state = True if heater_status == "ON" else False
                current_manual = control_data.get("manualState")

                # Update hanya jika berubah
                if current_manual != manual_state:
                    control_ref.set({
                        "manualState": manual_state
                    }, merge=True)    
        
    # SAVE TO FIRESTORE
    if pond_id:
        ai_ref = db.collection("ponds") \
            .document(pond_id) \
            .collection("ai")

        firestore_data = result.copy()
        timestamp = int(time.time() * 1000)
        firestore_data["timestamp"] = g_firestore.SERVER_TIMESTAMP 

        ai_ref.document(str(timestamp)).set(firestore_data)

    return jsonify(result)

@app.route('/predict-from-firestore/<pond_id>', methods=['GET'])
def predict_from_firestore(pond_id):

    docs = db.collection("ponds").document(pond_id)\
        .collection("realtime")\
        .order_by("timestamp", direction=g_firestore.Query.DESCENDING)\
        .limit(30).stream()

    sequence = []
    for doc in docs:
        d = doc.to_dict()
        sequence.append([
            float(d.get("suhu", 0)),
            float(d.get("pH", 7)),
            float(d.get("DO", 5))
        ])

    sequence = sequence[::-1]

    if len(sequence) < 30:
        return jsonify({"error": "Data belum cukup"}), 400

    return run_ai(np.array(sequence), pond_id)

# ===============================
# POST manual
# ===============================
@app.route("/predict", methods=["POST"])
def predict_manual():
    data = request.get_json()
    sequence = np.array(data["sequence"], dtype=float)
    pond_id = data.get("pond_id")   
    return run_ai(sequence, pond_id)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
