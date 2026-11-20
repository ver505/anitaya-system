import json
import os
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static", "images")
os.makedirs(STATIC_DIR, exist_ok=True)

METRICS_PATH = os.path.join(BASE_DIR, "models", "model_metrics.json")

with open(METRICS_PATH, "r", encoding="utf-8") as f:
    metrics = json.load(f)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"font.sans-serif": "Arial", "font.family": "sans-serif"})

# 1. 5-Model Comparison Chart (Validation CV R2 & Holdout Test MAE)
comp = metrics["model_comparison"]
models_sorted = sorted(comp.keys(), key=lambda m: comp[m].get("Rank", 99), reverse=True)
short_names = [
    m.replace(" Regressor", "").replace(" (Baseline)", "")
    for m in models_sorted
]
cv_r2_scores = [comp[m]["CV_R2_Mean"] for m in models_sorted]
test_r2_scores = [comp[m]["R2"] for m in models_sorted]
mae_mt_scores = [comp[m]["MAE"] / 10000.0 for m in models_sorted]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)

bar_colors = ["#059669" if comp[m].get("Is_Winner") else "#64748b" for m in models_sorted]
y_pos = list(range(len(models_sorted)))

bars_cv = ax1.barh([y + 0.17 for y in y_pos], cv_r2_scores, height=0.33, color="#0d9488", label="5-Fold CV R² (Train 2010–2021)")
bars_te = ax1.barh([y - 0.17 for y in y_pos], test_r2_scores, height=0.33, color="#94a3b8", label="Out-of-Time Test R² (2022–2025)")
ax1.set_yticks(y_pos)
ax1.set_yticklabels(short_names, fontsize=9.5, fontweight="bold")
ax1.set_xlim(0, 1.12)
ax1.set_xlabel("Coefficient of Determination ($R^2$)", fontsize=10.5, fontweight="bold")
ax1.set_title("AniTaya PH Dataset: 5-Model Validation vs. Out-of-Time $R^2$", fontsize=12, fontweight="bold", pad=12)
ax1.legend(loc="lower right", fontsize=8.5, frameon=True)
for bar in bars_cv:
    w = bar.get_width()
    ax1.text(w + 0.015, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center", fontsize=8.2, fontweight="bold", color="#0f766e")
for bar in bars_te:
    w = bar.get_width()
    ax1.text(w + 0.015, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center", fontsize=8.2, color="#475569")

bars2 = ax2.barh(y_pos, mae_mt_scores, color=bar_colors, height=0.55)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(short_names, fontsize=9.5, fontweight="bold")
ax2.set_xlabel("Holdout Test Mean Absolute Error (MT/ha — Lower is Better)", fontsize=10.5, fontweight="bold")
ax2.set_title("2022–2025 Out-of-Time Prediction Error (MAE in MT/ha)", fontsize=12, fontweight="bold", pad=12)
ax2.set_xlim(0, max(mae_mt_scores) * 1.18)
for bar in bars2:
    w = bar.get_width()
    ax2.text(w + 0.08, bar.get_y() + bar.get_height() / 2, f"{w:.2f} MT/ha", va="center", fontsize=8.5, fontweight="bold")

plt.tight_layout()
comparison_img = os.path.join(STATIC_DIR, "model_comparison.png")
plt.savefig(comparison_img)
plt.close()
print(f"Saved: {comparison_img}")

# 2. Feature Importance Chart (4 Philippine features, zero pesticide proxy)
feat_data = metrics["feature_importances"]
feat_names = [f["feature"].split(" (")[0] for f in feat_data]
feat_vals = [f["importance"] for f in feat_data]

fig, ax = plt.subplots(figsize=(9, 4.8), dpi=300)
colors_fi = ["#10b981", "#3b82f6", "#f59e0b", "#8b5cf6"]
bars = ax.barh(feat_names[::-1], feat_vals[::-1], color=colors_fi[:len(feat_names)][::-1], height=0.5)
ax.set_xlabel("Relative Importance Percentage (%)", fontsize=11, fontweight="bold")
ax.set_title(f"AniTaya ({metrics['best_model']['name']}): Feature Importance", fontsize=12.5, fontweight="bold", pad=12)
for bar in bars:
    w = bar.get_width()
    ax.text(w + 1.0, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=10, fontweight="bold")
ax.set_xlim(0, max(feat_vals) * 1.18)
plt.tight_layout()
feat_img = os.path.join(STATIC_DIR, "feature_importance.png")
plt.savefig(feat_img)
plt.close()
print(f"Saved: {feat_img}")

# 3. Actual vs Predicted Chart (Strictly Unseen 2022-2025 Holdout Test Samples)
avp = metrics["actual_vs_predicted"]
y_sample = [p["actual_mt_ha"] for p in avp]
y_pred_sample = [p["predicted_mt_ha"] for p in avp]

fig, ax = plt.subplots(figsize=(7.5, 6), dpi=300)
ax.scatter(y_sample, y_pred_sample, alpha=0.5, color="#059669", edgecolors="none", s=34)
max_lim = max(max(y_sample), max(y_pred_sample)) * 1.05
lims = [0, max_lim]
ax.plot(lims, lims, color="#dc2626", linestyle="--", linewidth=1.5, label="Perfect Agreement (1:1)")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("Actual PSA Provincial Harvest Yield (MT / Hectare)", fontsize=11, fontweight="bold")
ax.set_ylabel("Predicted Yield Rate (MT / Hectare)", fontsize=11, fontweight="bold")
ax.set_title(
    f"Unseen 2022–2025 Holdout Test: Actual vs. Predicted ({metrics['best_model']['name']})",
    fontsize=12,
    fontweight="bold",
    pad=12,
)
ax.legend(frameon=True)
plt.tight_layout()
avp_img = os.path.join(STATIC_DIR, "actual_vs_predicted.png")
plt.savefig(avp_img)
plt.close()
print(f"Saved: {avp_img}")
print("All 5-model Philippine evaluation charts generated cleanly!")
