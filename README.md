# 🐟 SMIKOLE-AI

> Sistem Monitoring dan Prediksi Kualitas Kolam Lele berbasis Machine Learning

SMIKOLE-AI adalah platform AI untuk tambak lele yang menggabungkan dua model machine learning:
1. **LSTM** — untuk memprediksi kualitas air secara real-time (suhu, pH, DO)
2. **Random Forest** — untuk memprediksi pertumbuhan ikan dan efisiensi pakan (FCR/ADG)

Keduanya terintegrasi dengan **Google Firestore** dan dapat di-deploy menggunakan **Docker**.

---

## 📁 Struktur Proyek

```
SMIKOLE-AI/
├── ml-service/                   # Layanan prediksi kualitas air (LSTM)
│   ├── app.py                    # Flask API utama
│   ├── train_model.py            # Script pelatihan model LSTM
│   ├── water_quality_data.csv    # Dataset kualitas air
│   ├── lstm_water_quality.h5     # Model LSTM terlatih
│   ├── lstm_water_quality_best.h5# Model LSTM terbaik (checkpoint)
│   ├── scaler_water_quality.pkl  # MinMaxScaler tersimpan
│   ├── requirements.txt          # Dependensi Python
│   ├── Dockerfile                # Konfigurasi Docker
│   ├── training_history.png      # Grafik riwayat pelatihan
│   └── prediction_vs_actual.png  # Grafik prediksi vs aktual
│
└── ml-service RF/                # Layanan prediksi pertumbuhan (Random Forest)
    ├── app.py                    # Flask API utama
    ├── predict.py                # Logika prediksi & rekomendasi
    ├── training.py               # Script pelatihan model RF
    ├── generate_dataset.py       # Generator dataset sintetis
    ├── evaluation_report.py      # Laporan evaluasi model
    ├── generate_charts.py        # Generator grafik evaluasi
    ├── model_growth.pkl          # Model Random Forest terlatih
    ├── dataset.csv               # Dataset pertumbuhan lele
    ├── requirements.txt          # Dependensi Python
    ├── Dockerfile                # Konfigurasi Docker
    └── chart_*.png               # Grafik evaluasi model
```

---

## 🧠 Model 1 — LSTM: Prediksi Kualitas Air

### Deskripsi

Model **Long Short-Term Memory (LSTM)** digunakan untuk memprediksi nilai kualitas air **30 menit ke depan** berdasarkan 30 data pembacaan sensor terakhir.

### Parameter yang Diprediksi

| Parameter | Rentang Normal | Satuan |
|-----------|---------------|--------|
| Suhu (Temperature) | 25.0 – 30.0 | °C |
| Derajat Keasaman (pH) | 7.0 – 8.5 | — |
| Oksigen Terlarut (DO) | ≥ 3.0 | mg/L |

### Arsitektur Model

```
Input (30 timesteps × 3 fitur)
  └─► LSTM(64, return_sequences=True)
        └─► Dropout(0.2)
              └─► LSTM(32)
                    └─► Dropout(0.2)
                          └─► Dense(16, relu)
                                └─► Dense(3) → [Temp, pH, DO]
```

### Cara Melatih Model

```bash
cd ml-service
pip install -r requirements.txt
python train_model.py
```

### Menjalankan API (LSTM)

```bash
cd ml-service
export API_KEY="your_api_key"
python app.py
# Server berjalan di http://localhost:8080
```

### Endpoint API

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| `GET` | `/health` | Cek status layanan |
| `GET` | `/predict-from-firestore/<pond_id>` | Prediksi otomatis dari data Firestore |
| `POST` | `/predict` | Prediksi manual dengan data JSON |

**Contoh Request POST `/predict`:**
```json
{
  "sequence": [
    [27.5, 7.2, 5.1],
    [27.6, 7.3, 5.0],
    ...
  ],
  "pond_id": "kolam_01"
}
```

**Contoh Response:**
```json
{
  "predicted_temperature_30min": 27.8,
  "predicted_ph_30min": 7.25,
  "predicted_do_30min": 4.95,
  "heater_status": "OFF",
  "temperature_status": "normal",
  "risk_status": "normal",
  "recommendations": [
    "✅ Semua parameter normal dan stabil. Sistem berjalan baik."
  ]
}
```

---

## 🌱 Model 2 — Random Forest: Prediksi Pertumbuhan Lele

### Deskripsi

Model **Random Forest Regressor** memprediksi pertambahan bobot lele per ekor selama periode tertentu, sekaligus menghitung metrik penting seperti **ADG** (Average Daily Gain) dan **FCR** (Feed Conversion Ratio).

### Input Prediksi

| Parameter | Tipe | Deskripsi |
|-----------|------|-----------|
| `siklus` | int | Siklus budidaya ke-n |
| `DOC` | int | Day of Culture (hari ke berapa) |
| `populasi` | int | Jumlah ikan (ekor) |
| `bobot_awal_per_ekor_gr` | float | Bobot awal per ekor (gram) |
| `pakan_harian_gr` | float | Pakan harian total (gram) |
| `panjang_periode_hari` | int | Durasi periode prediksi (hari) |

### Output & Metrik

| Metrik | Deskripsi |
|--------|-----------|
| `ADG` | Average Daily Gain — pertambahan bobot rata-rata per ekor per hari |
| `FCR` | Feed Conversion Ratio — efisiensi konversi pakan |
| `status_efisiensi` | `Efisien` / `Normal` / `Overfeeding` |
| `biomassa_akhir_kg` | Estimasi total biomassa akhir periode |
| `rasio_pakan_next_persen` | Rekomendasi rasio pakan untuk periode berikutnya |

### Cara Melatih Model

```bash
cd "ml-service RF"
pip install -r requirements.txt
python generate_dataset.py  # Generate dataset
python training.py          # Latih model
python evaluation_report.py # Evaluasi performa
python generate_charts.py   # Buat grafik evaluasi
```

### Menjalankan API (Random Forest)

```bash
cd "ml-service RF"
python app.py
# Server berjalan di http://localhost:5000
```

### Endpoint API

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| `GET` | `/health` | Cek status layanan |
| `POST` | `/predict` | Prediksi pertumbuhan & rekomendasi pakan |
| `GET` | `/fcr` | Riwayat prediksi dari Firestore |
| `GET` | `/export/csv` | Export riwayat sebagai file CSV |

**Contoh Request POST `/predict`:**
```json
{
  "siklus": 1,
  "DOC": 30,
  "populasi": 2000,
  "bobot_awal_per_ekor_gr": 15.0,
  "pakan_harian_gr": 1500,
  "panjang_periode_hari": 10
}
```

**Contoh Response:**
```json
{
  "input": { "siklus": 1, "DOC": 30, ... },
  "metrics": {
    "ADG_gr_per_ekor_hari": 0.85,
    "FCR": 1.12,
    "status_efisiensi": "Normal"
  },
  "predictions": {
    "delta_bobot_per_ekor_gr": 8.5,
    "bobot_akhir_per_ekor_gr": 23.5,
    "biomassa_awal_kg": 30.0,
    "biomassa_akhir_kg": 47.0
  },
  "recommendations": {
    "rasio_pakan_next_persen": 5,
    "pakan_harian_next_gr": 2350,
    "catatan_rasio": "Rasio stabil fase pertumbuhan utama",
    "aksi": "Efisiensi pakan normal, monitor terus"
  }
}
```

---

## 🐳 Deployment dengan Docker

### Build & Run — LSTM Service

```bash
cd ml-service
docker build -t smikole-lstm .
docker run -p 8080:8080 \
  -e API_KEY="your_api_key" \
  -e GOOGLE_APPLICATION_CREDENTIALS="/app/credentials.json" \
  smikole-lstm
```

### Build & Run — Random Forest Service

```bash
cd "ml-service RF"
docker build -t smikole-rf .
docker run -p 5000:5000 \
  -e GOOGLE_APPLICATION_CREDENTIALS="/app/credentials.json" \
  smikole-rf
```

---

## ☁️ Integrasi Google Cloud

Kedua layanan menggunakan **Google Firestore** untuk:
- Menyimpan hasil prediksi secara otomatis
- Membaca data sensor real-time dari kolam
- Mengontrol perangkat (heater) secara otomatis dalam mode `AUTO`

**Autentikasi:**  
Gunakan **Application Default Credentials (ADC)** atau sediakan file service account JSON melalui environment variable `GOOGLE_APPLICATION_CREDENTIALS`.

---

## 📦 Dependensi Utama

### LSTM Service
- `TensorFlow / Keras` — model deep learning
- `Flask` — web framework API
- `scikit-learn` — normalisasi data (MinMaxScaler)
- `firebase-admin` + `google-cloud-firestore` — database

### Random Forest Service
- `scikit-learn` — model Random Forest
- `Flask` + `Flask-CORS` — web framework API
- `pandas` + `numpy` — manipulasi data
- `google-cloud-firestore` — database

---

## 📊 Evaluasi Model

### LSTM — Kualitas Air
Model dievaluasi menggunakan **MSE** dan **MAE** pada data test. Grafik evaluasi tersedia di:
- `ml-service/training_history.png` — kurva loss pelatihan
- `ml-service/prediction_vs_actual.png` — prediksi vs nilai aktual

### Random Forest — Pertumbuhan
Grafik evaluasi model tersedia di `ml-service RF/`:
- `chart_feature_importance.png` — pentingnya setiap fitur
- `chart_scatter_pred_vs_actual.png` — scatter plot prediksi vs aktual
- `chart_residual_plot.png` — analisis residual
- `chart_error_distribution.png` — distribusi error
- `chart_train_vs_test.png` — perbandingan train dan test

---

## 🔐 Keamanan

Layanan LSTM dilindungi dengan **API Key** yang dikirim melalui header:
```
x-api-key: your_api_key
```

Endpoint `/health` dikecualikan dari autentikasi untuk keperluan health check.

---

## 📄 Lisensi

Proyek ini dikembangkan untuk keperluan sistem monitoring tambak lele cerdas SMIKOLE.
