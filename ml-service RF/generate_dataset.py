"""
Generate realistic catfish (lele) growth dataset based on per-fish metrics.
Features: DOC, populasi, bobot_awal_per_ekor_gr, pakan_harian_gr, panjang_periode_hari
Target: delta_bobot_per_ekor_gr
"""
import numpy as np
import pandas as pd

np.random.seed(42)
records = []

# ===== REAL DATA CONVERTED TO PER-FISH METRICS =====
# Siklus 1
records.append({'DOC': 11, 'populasi': 2092, 'bobot_awal_per_ekor_gr': 1.5, 'pakan_harian_gr': 160, 'panjang_periode_hari': 9, 'delta_bobot_per_ekor_gr': 1.34})
records.append({'DOC': 20, 'populasi': 2000, 'bobot_awal_per_ekor_gr': 2.86, 'pakan_harian_gr': 180, 'panjang_periode_hari': 9, 'delta_bobot_per_ekor_gr': 2.72}) # Adjusted from total delta 3509
records.append({'DOC': 30, 'populasi': 1900, 'bobot_awal_per_ekor_gr': 5.58, 'pakan_harian_gr': 493, 'panjang_periode_hari': 10, 'delta_bobot_per_ekor_gr': 1.37}) # Adjusted from delta 2612
records.append({'DOC': 50, 'populasi': 422, 'bobot_awal_per_ekor_gr': 26.0, 'pakan_harian_gr': 573, 'panjang_periode_hari': 10, 'delta_bobot_per_ekor_gr': 19.46}) # Siklus 1 >10
records.append({'DOC': 50, 'populasi': 382, 'bobot_awal_per_ekor_gr': 19.2, 'pakan_harian_gr': 476, 'panjang_periode_hari': 10, 'delta_bobot_per_ekor_gr': 12.05}) # Siklus 1 <10

# Siklus 3
records.append({'DOC': 18, 'populasi': 4207, 'bobot_awal_per_ekor_gr': 0.63, 'pakan_harian_gr': 319, 'panjang_periode_hari': 10, 'delta_bobot_per_ekor_gr': 1.07})
records.append({'DOC': 26, 'populasi': 4150, 'bobot_awal_per_ekor_gr': 1.7, 'pakan_harian_gr': 503, 'panjang_periode_hari': 8, 'delta_bobot_per_ekor_gr': 0.83}) # Adjusted
records.append({'DOC': 38, 'populasi': 2953, 'bobot_awal_per_ekor_gr': 2.8, 'pakan_harian_gr': 216, 'panjang_periode_hari': 12, 'delta_bobot_per_ekor_gr': 0.91}) # <8
records.append({'DOC': 38, 'populasi': 253, 'bobot_awal_per_ekor_gr': 9.05, 'pakan_harian_gr': 586, 'panjang_periode_hari': 12, 'delta_bobot_per_ekor_gr': 19.97}) # >8
records.append({'DOC': 60, 'populasi': 3000, 'bobot_awal_per_ekor_gr': 4.46, 'pakan_harian_gr': 1465, 'panjang_periode_hari': 22, 'delta_bobot_per_ekor_gr': 6.26})


print(f"Real data points: {len(records)}")

# ===== SYNTHETIC DATA GENERATION =====
def simulate_cycle(start_pop, start_weight_per_fish, seed_offset=0, n_periods=9, period_days=10):
    rng = np.random.RandomState(42 + seed_offset)
    cycle_records = []
    populasi = start_pop
    bobot_ekor = start_weight_per_fish

    for period in range(n_periods):
        doc = (period + 1) * period_days
        
        # Simulate slight mortality (0-2% per period)
        survival_rate = rng.uniform(0.98, 1.0)
        populasi = int(populasi * survival_rate)

        # Biomass
        biomassa = populasi * bobot_ekor

        # Feed ratio based on DOC (farmer practice)
        if doc <= 10:
            rasio = rng.choice([3, 4, 5])
        elif doc <= 40:
            rasio = 5
        elif doc <= 70:
            rasio = rng.choice([5, 6])
        else:
            rasio = 6

        daily_pakan_total = biomassa * rasio / 100
        total_pakan = daily_pakan_total * period_days

        # FCR varies by growth phase (calibrated from farmer data)
        if doc <= 10:
            fcr = rng.uniform(1.2, 2.5)
        elif doc <= 30:
            fcr = rng.uniform(0.7, 0.95)
        elif doc <= 50:
            fcr = rng.uniform(0.75, 1.05)
        elif doc <= 70:
            fcr = rng.uniform(0.85, 1.15)
        else:
            fcr = rng.uniform(0.90, 1.30)

        fcr *= rng.uniform(0.95, 1.05)  # noise
        
        delta_m_total = total_pakan / fcr
        delta_m_ekor = delta_m_total / populasi

        cycle_records.append({
            'DOC': doc,
            'populasi': populasi,
            'bobot_awal_per_ekor_gr': round(bobot_ekor, 2),
            'pakan_harian_gr': round(daily_pakan_total),
            'panjang_periode_hari': period_days,
            'delta_bobot_per_ekor_gr': round(delta_m_ekor, 2)
        })

        bobot_ekor += delta_m_ekor

    return cycle_records

# 20 simulated cycles with varied starting configs
synthetic_configs = [
    (2000, 1.5), (2500, 2.0), (3000, 1.2), (4000, 0.8), (1500, 3.0),
    (2100, 1.5), (2600, 1.8), (3100, 1.4), (4200, 0.7), (1800, 2.5),
    (2200, 1.6), (2700, 1.9), (3200, 1.3), (4500, 0.6), (1900, 2.8),
    (2300, 1.7), (2800, 2.1), (3500, 1.1), (5000, 0.5), (2400, 2.2)
]

for i, (start_pop, start_weight) in enumerate(synthetic_configs):
    cycle = simulate_cycle(start_pop, start_weight, seed_offset=i)
    records.extend(cycle)

# Build DataFrame and save
df = pd.DataFrame(records)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv('dataset.csv', index=False)

print(f"Total records: {len(df)}")
print("\n=== Dataset Statistics ===")
print(df.describe().round(2))
print(f"\nSaved new format to dataset.csv")
