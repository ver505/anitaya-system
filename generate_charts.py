import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static", "images")
os.makedirs(STATIC_DIR, exist_ok=True)

METRICS_PATH = os.path.join(BASE_DIR, "models", "model_metrics.json")
MODEL_PATH = os.path.join(BASE_DIR, "models", "anitaya_best_pipeline.joblib")
DATA_PATH = os.path.join(BASE_DIR, "data", "df_yield_merged.csv")

with open(METRICS_PATH, "r", encoding="utf-8") as f:
    metrics = json.load(f)

# Set clean aesthetic style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"font.sans-serif": "Arial", "font.family": "sans-serif"})

# 1. Model Comparison Chart (R2 & MAE)
models = list(metrics["model_comparison"].keys())
short_names = [m.replace(" Regressor", "").replace(" (Primary Model)", "").replace(" (Baseline)", "") for m in models]
r2_scores = [metrics["model_comparison"][m]["R2"] for m in models]
mae_scores = [metrics["model_comparison"][m]["MAE"] / 1000 for m in models] # in k hg/ha

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

colors = ["#94a3b8", "#94a3b8", "#64748b", "#059669", "#0d9488"]

bars1 = ax1.barh(short_names, r2_scores, color=colors, height=0.55, edgecolor="none")
ax1.set_xlim(0, 1.05)
ax1.set_xlabel("Coefficient of Determination ($R^2$ Score)", fontsize=11, fontweight="bold")
ax1.set_title("CSST 102: Model Accuracy Comparison ($R^2$)", fontsize=13, fontweight="bold", pad=12)
for bar in bars1:
    w = bar.get_width()
    ax1.text(w + 0.02, bar.get_y() + bar.get_height()/2, f"{w:.4f}", va="center", fontsize=10, fontweight="bold")

bars2 = ax2.barh(short_names, mae_scores, color=colors, height=0.55, edgecolor="none")
ax2.set_xlabel("Mean Absolute Error (in thousands hg/ha)", fontsize=11, fontweight="bold")
ax2.set_title("Prediction Error Comparison (MAE - Lower is Better)", fontsize=13, fontweight="bold", pad=12)
for bar in bars2:
    w = bar.get_width()
    ax2.text(w + 0.8, bar.get_y() + bar.get_height()/2, f"{w:.1f}k", va="center", fontsize=10, fontweight="bold")

plt.tight_layout()
comparison_img = os.path.join(STATIC_DIR, "model_comparison.png")
plt.savefig(comparison_img)
plt.close()
print(f"Saved: {comparison_img}")

# 2. Feature Importance Chart
feat_data = metrics["feature_importances"]
feat_names = [f["feature"].split(" (")[0] for f in feat_data]
feat_vals = [f["importance"] for f in feat_data]

fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
colors_fi = ["#10b981", "#3b82f6", "#f59e0b", "#ec4899", "#8b5cf6"]
bars = ax.barh(feat_names[::-1], feat_vals[::-1], color=colors_fi[:len(feat_names)][::-1], height=0.55)
ax.set_xlabel("Relative Importance Percentage (%)", fontsize=11, fontweight="bold")
ax.set_title("AniTaya: Feature Importance in Harvest Yield Prediction", fontsize=13, fontweight="bold", pad=12)
for bar in bars:
    w = bar.get_width()
    ax.text(w + 1.0, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va="center", fontsize=10, fontweight="bold")
ax.set_xlim(0, max(feat_vals) * 1.15)
plt.tight_layout()
feat_img = os.path.join(STATIC_DIR, "feature_importance.png")
plt.savefig(feat_img)
plt.close()
print(f"Saved: {feat_img}")

# 3. Actual vs Predicted Sample
df = pd.read_csv(DATA_PATH).sample(1000, random_state=42)
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])
pipeline = joblib.load(MODEL_PATH)

feature_cols = ["item", "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp", "area"]
X_sample = df[feature_cols]
y_sample = df["value_hg_ha"] / 10000 # Convert to Metric Tons / ha
y_pred_sample = pipeline.predict(X_sample) / 10000

fig, ax = plt.subplots(figsize=(7.5, 6), dpi=300)
ax.scatter(y_sample, y_pred_sample, alpha=0.35, color="#059669", edgecolors="none", s=28)
lims = [0, max(y_sample.max(), y_pred_sample.max()) * 1.05]
ax.plot(lims, lims, color="#dc2626", linestyle="--", linewidth=1.5, label="Perfect Agreement (1:1)")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("Actual Harvest Yield (Metric Tons / Hectare)", fontsize=11, fontweight="bold")
ax.set_ylabel("Predicted Harvest Yield (Metric Tons / Hectare)", fontsize=11, fontweight="bold")
ax.set_title("AniTaya: Actual vs. Predicted Crop Yield (Test Subset)", fontsize=13, fontweight="bold", pad=12)
ax.legend(frameon=True)
plt.tight_layout()
avp_img = os.path.join(STATIC_DIR, "actual_vs_predicted.png")
plt.savefig(avp_img)
plt.close()
print(f"Saved: {avp_img}")
print("All static evaluation charts successfully generated!")
