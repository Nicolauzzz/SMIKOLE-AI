"""
Prediction module with decision logic for catfish growth (Per-fish metrics).
Loads trained model and provides full recommendation output including ADG and FCR.
"""
import joblib
import pandas as pd
import os

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_growth.pkl")

def load_model():
    return joblib.load(MODEL_PATH)

def get_feed_ratio_recommendation(doc):
    """Rule-based feed ratio recommendation based on DOC (farmer practice)."""
    if doc <= 3:
        return 3, "Fase awal, rasio rendah untuk adaptasi"
    elif doc <= 5:
        return 4, "Fase awal, naikkan rasio bertahap"
    elif doc <= 7:
        return 5, "Fase awal, mendekati rasio normal"
    elif doc <= 10:
        return 7, "Golden age - masa pertumbuhan krusial"
    elif doc <= 40:
        return 5, "Rasio stabil fase pertumbuhan utama"
    elif doc <= 70:
        return 5, "Rasio stabil fase pertumbuhan lanjut"
    else:
        return 6, "Fase akhir, naikkan sedikit untuk finishing"

def predict_growth(siklus, DOC, populasi, bobot_awal_per_ekor_gr, pakan_harian_gr, panjang_periode_hari):
    """
    Predict fish growth and provide full recommendation based on per-fish metrics.
    """
    model = load_model()
    
    input_data = pd.DataFrame([{
        'DOC': DOC,
        'populasi': populasi,
        'bobot_awal_per_ekor_gr': bobot_awal_per_ekor_gr,
        'pakan_harian_gr': pakan_harian_gr,
        'panjang_periode_hari': panjang_periode_hari
    }])
    
    predicted_delta_bobot_ekor = float(model.predict(input_data)[0])
    
    # 1. Biomass calculations
    biomassa_awal_total_gr = populasi * bobot_awal_per_ekor_gr
    bobot_akhir_per_ekor_gr = bobot_awal_per_ekor_gr + predicted_delta_bobot_ekor
    biomassa_akhir_total_gr = populasi * bobot_akhir_per_ekor_gr
    
    delta_biomassa_total_gr = biomassa_akhir_total_gr - biomassa_awal_total_gr
    total_pakan_periode_gr = pakan_harian_gr * panjang_periode_hari

    # 2. Key Metrics (ADG & FCR)
    adg_gr_per_ekor_hari = predicted_delta_bobot_ekor / panjang_periode_hari if panjang_periode_hari > 0 else 0
    
    if delta_biomassa_total_gr > 0:
        fcr = total_pakan_periode_gr / delta_biomassa_total_gr
    else:
        fcr = float('inf')
        
    # 3. Recommendations
    recommended_rasio, rasio_note = get_feed_ratio_recommendation(DOC + panjang_periode_hari)
    rekomendasi_pakan_harian_next_gr = biomassa_akhir_total_gr * recommended_rasio / 100
    
    # 4. Decision logic
    if fcr < 1.0:
        status_efisiensi = "Efisien"
        rekomendasi_aksi = "Pakan optimal, pertahankan rasio"
    elif fcr <= 1.3:
        status_efisiensi = "Normal"
        rekomendasi_aksi = "Efisiensi pakan normal, monitor terus"
    else:
        status_efisiensi = "Overfeeding"
        rekomendasi_aksi = "Kurangi rasio pakan, FCR terlalu tinggi"
        
    return {
        'input': {
            'siklus': siklus,
            'DOC': DOC,
            'populasi': populasi,
            'bobot_awal_per_ekor_gr': float(bobot_awal_per_ekor_gr),
            'pakan_harian_gr': float(pakan_harian_gr),
            'panjang_periode_hari': panjang_periode_hari
        },
        'metrics': {
            'ADG_gr_per_ekor_hari': round(adg_gr_per_ekor_hari, 2),
            'FCR': round(fcr, 2),
            'status_efisiensi': status_efisiensi
        },
        'predictions': {
            'delta_bobot_per_ekor_gr': round(predicted_delta_bobot_ekor, 2),
            'bobot_akhir_per_ekor_gr': round(bobot_akhir_per_ekor_gr, 2),
            'biomassa_awal_kg': round(biomassa_awal_total_gr / 1000, 2),
            'biomassa_akhir_kg': round(biomassa_akhir_total_gr / 1000, 2)
        },
        'recommendations': {
            'rasio_pakan_next_persen': recommended_rasio,
            'pakan_harian_next_gr': round(rekomendasi_pakan_harian_next_gr),
            'catatan_rasio': rasio_note,
            'aksi': rekomendasi_aksi
        }
    }

if __name__ == "__main__":
    # Test
    print("=== Test Prediction (Excel Overhaul) ===")
    res = predict_growth(siklus=1, DOC=30, populasi=2000, bobot_awal_per_ekor_gr=15.0, pakan_harian_gr=1500, panjang_periode_hari=10)
    import json
    print(json.dumps(res, indent=2))
