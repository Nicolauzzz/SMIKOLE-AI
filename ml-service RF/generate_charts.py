"""
Generate evaluation charts for Random Forest model report.
Outputs: Feature Importance, Scatter Plot, Residual Plot, Error Distribution.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ============================================================
# Setup: Reproduce exact model
# ============================================================
data = pd.read_csv("dataset.csv")
features = ["DOC", "populasi", "bobot_awal_per_ekor_gr", "pakan_harian_gr", "panjang_periode_hari"]
X = data[features]
y = data["delta_bobot_per_ekor_gr"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(
    n_estimators=200, max_depth=12,
    min_samples_split=5, min_samples_leaf=2,
    random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2 = r2_score(y_test, y_pred_test)

# ============================================================
# Style Configuration
# ============================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'figure.facecolor': 'white',
    'axes.facecolor': '#FAFAFA',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

# Color palette
COLORS = {
    'primary': '#2563EB',
    'secondary': '#10B981',
    'accent': '#F59E0B',
    'danger': '#EF4444',
    'purple': '#8B5CF6',
    'gray': '#6B7280',
    'light_blue': '#DBEAFE',
    'light_green': '#D1FAE5',
}

# ============================================================
# Chart 1: Feature Importance Bar Chart
# ============================================================
print("Generating Chart 1: Feature Importance...")

fig, ax = plt.subplots(figsize=(10, 6))

feat_names_display = {
    'bobot_awal_per_ekor_gr': 'Bobot Awal\n(gr/ekor)',
    'pakan_harian_gr': 'Pakan Harian\n(gr)',
    'populasi': 'Populasi\n(ekor)',
    'DOC': 'DOC\n(hari)',
    'panjang_periode_hari': 'Panjang Periode\n(hari)'
}

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
sorted_features = [features[i] for i in indices]
sorted_importances = importances[indices]
sorted_labels = [feat_names_display[f] for f in sorted_features]

# Gradient colors based on importance
bar_colors = [COLORS['primary'] if imp > 0.5 else 
              COLORS['secondary'] if imp > 0.01 else 
              COLORS['gray'] for imp in sorted_importances]

bars = ax.bar(range(len(sorted_features)), sorted_importances, 
              color=bar_colors, edgecolor='white', linewidth=1.5,
              width=0.6, zorder=3)

# Add value labels on bars
for bar, imp in zip(bars, sorted_importances):
    if imp > 0.05:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{imp*100:.2f}%', ha='center', va='bottom', fontweight='bold',
                fontsize=12, color=COLORS['primary'])
    else:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{imp*100:.2f}%', ha='center', va='bottom', fontsize=10,
                color=COLORS['gray'])

ax.set_xticks(range(len(sorted_features)))
ax.set_xticklabels(sorted_labels, fontsize=10)
ax.set_ylabel('Importance Score')
ax.set_title('Feature Importance — Random Forest Regressor')
ax.set_ylim(0, max(sorted_importances) * 1.15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('chart_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] Saved: chart_feature_importance.png")

# ============================================================
# Chart 2: Scatter Plot — Prediksi vs Aktual
# ============================================================
print("Generating Chart 2: Scatter Plot Prediksi vs Aktual...")

fig, ax = plt.subplots(figsize=(8, 8))

# Plot train data (lighter)
ax.scatter(y_train, y_pred_train, alpha=0.4, s=40, c=COLORS['light_blue'],
           edgecolors=COLORS['primary'], linewidth=0.5, label=f'Training (n={len(y_train)})', zorder=2)

# Plot test data (bolder)
ax.scatter(y_test, y_pred_test, alpha=0.8, s=60, c=COLORS['secondary'],
           edgecolors='white', linewidth=0.8, label=f'Testing (n={len(y_test)})', zorder=3)

# Perfect prediction line
max_val = max(y.max(), max(y_pred_train.max(), y_pred_test.max())) * 1.05
ax.plot([0, max_val], [0, max_val], 'r--', linewidth=1.5, alpha=0.7, 
        label='Prediksi Sempurna', zorder=4)

# Metrics annotation
textstr = f'Test Metrics:\nMAE = {mae:.4f} gr\nRMSE = {rmse:.4f} gr\nR² = {r2:.4f}'
props = dict(boxstyle='round,pad=0.5', facecolor=COLORS['light_green'], alpha=0.8, edgecolor=COLORS['secondary'])
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

ax.set_xlabel('Nilai Aktual (gram/ekor)')
ax.set_ylabel('Nilai Prediksi (gram/ekor)')
ax.set_title('Scatter Plot — Prediksi vs Aktual Delta Bobot')
ax.set_xlim(0, max_val)
ax.set_ylim(0, max_val)
ax.set_aspect('equal')
ax.legend(loc='lower right', framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('chart_scatter_pred_vs_actual.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] Saved: chart_scatter_pred_vs_actual.png")

# ============================================================
# Chart 3: Residual Plot (Error Distribution)
# ============================================================
print("Generating Chart 3: Residual Plot...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

residuals = y_test.values - y_pred_test

# Left: Residuals vs Predicted
ax1.scatter(y_pred_test, residuals, alpha=0.7, s=50, c=COLORS['primary'],
            edgecolors='white', linewidth=0.5, zorder=3)
ax1.axhline(y=0, color=COLORS['danger'], linestyle='--', linewidth=1.5, alpha=0.7)
ax1.axhline(y=mae, color=COLORS['accent'], linestyle=':', linewidth=1, alpha=0.7, label=f'+MAE ({mae:.2f})')
ax1.axhline(y=-mae, color=COLORS['accent'], linestyle=':', linewidth=1, alpha=0.7, label=f'-MAE ({mae:.2f})')
ax1.set_xlabel('Nilai Prediksi (gram/ekor)')
ax1.set_ylabel('Residual (Aktual - Prediksi)')
ax1.set_title('Residual vs Prediksi')
ax1.legend(fontsize=9)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Right: Histogram of residuals
ax2.hist(residuals, bins=15, color=COLORS['primary'], edgecolor='white',
         linewidth=1, alpha=0.8, zorder=3)
ax2.axvline(x=0, color=COLORS['danger'], linestyle='--', linewidth=1.5, alpha=0.7, label='Zero Error')
ax2.axvline(x=np.mean(residuals), color=COLORS['accent'], linestyle='-', linewidth=2,
            label=f'Mean = {np.mean(residuals):.3f}')
ax2.set_xlabel('Residual (gram/ekor)')
ax2.set_ylabel('Frekuensi')
ax2.set_title('Distribusi Residual')
ax2.legend(fontsize=9)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('chart_residual_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] Saved: chart_residual_plot.png")

# ============================================================
# Chart 4: Error Distribution Bar Chart
# ============================================================
print("Generating Chart 4: Error Distribution...")

fig, ax = plt.subplots(figsize=(8, 5))

errors = abs(y_test.values - y_pred_test)
categories = ['< 0,5 gram', '0,5 - 1,0 gram', '1,0 - 2,0 gram', '>= 2,0 gram']
counts = [
    (errors < 0.5).sum(),
    ((errors >= 0.5) & (errors < 1.0)).sum(),
    ((errors >= 1.0) & (errors < 2.0)).sum(),
    (errors >= 2.0).sum()
]
percentages = [c / len(errors) * 100 for c in counts]

bar_colors = [COLORS['secondary'], COLORS['primary'], COLORS['accent'], COLORS['danger']]
bars = ax.bar(categories, counts, color=bar_colors, edgecolor='white', linewidth=1.5, width=0.6, zorder=3)

for bar, count, pct in zip(bars, counts, percentages):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{count} ({pct:.1f}%)', ha='center', va='bottom',
            fontweight='bold', fontsize=11)

ax.set_ylabel('Jumlah Sampel')
ax.set_title(f'Distribusi Error Prediksi (n={len(errors)} sampel test)')
ax.set_ylim(0, max(counts) * 1.25)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('chart_error_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] Saved: chart_error_distribution.png")

# ============================================================
# Chart 5: Perbandingan Training vs Testing Metrics
# ============================================================
print("Generating Chart 5: Training vs Testing Comparison...")

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

metrics_names = ['MAE\n(gram/ekor)', 'RMSE\n(gram/ekor)', 'R2 Score']
train_vals = [
    mean_absolute_error(y_train, y_pred_train),
    np.sqrt(mean_squared_error(y_train, y_pred_train)),
    r2_score(y_train, y_pred_train)
]
test_vals = [mae, rmse, r2]

for i, (ax, name, tv, tsv) in enumerate(zip(axes, metrics_names, train_vals, test_vals)):
    x = np.arange(2)
    bars = ax.bar(x, [tv, tsv], color=[COLORS['primary'], COLORS['secondary']],
                  edgecolor='white', linewidth=1.5, width=0.5, zorder=3)
    
    for bar, val in zip(bars, [tv, tsv]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_xticks(x)
    ax.set_xticklabels(['Training', 'Testing'], fontsize=11)
    ax.set_title(name, fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Adjust y-axis
    if i < 2:  # MAE, RMSE
        ax.set_ylim(0, max(tv, tsv) * 1.3)
    else:  # R²
        ax.set_ylim(0.95, 1.0)

plt.suptitle('Perbandingan Metrik Training vs Testing', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('chart_train_vs_test.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] Saved: chart_train_vs_test.png")

# ============================================================
print(f"\n{'='*50}")
print("[OK] Semua grafik berhasil di-generate!")
print(f"{'='*50}")
print("Files:")
print("  1. chart_feature_importance.png")
print("  2. chart_scatter_pred_vs_actual.png")
print("  3. chart_residual_plot.png")
print("  4. chart_error_distribution.png")
print("  5. chart_train_vs_test.png")
