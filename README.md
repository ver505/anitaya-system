# AniTaya – Philippine Crop Recommendation, Harvest Prediction & Income Estimation System
**Web-Based Predictive Machine Learning Decision-Support System for Filipino Farmers**

**Institution:** Laguna State Polytechnic University (LSPU) – College of Computer Studies  
**Course:** CSST 102 – Basic Machine Learning | 1st Semester AY 2026–2027  

---

## 1. System Overview & Philippine Integration

**AniTaya** (*Ani* = Harvest, *Taya* = Forecast/Estimate) forecasts Philippine crop harvest yields and net farm income across **all 82 Philippine provinces** and **35 crops**, eliminating legacy global proxies (`Indonesia`, hardcoded `pesticides_tonnes=120.0`) in favor of real Philippine public datasets:

1. **PSA OpenSTAT Provincial Crop Production & Harvested Area (`DB/2E/CS`, 1987–2025):** **36,088 clean records** across 82 provinces and 35 crops (`data/psa_provincial_production.csv`), converting $\text{MT} / \text{HA} \times 10{,}000$ to $\text{hg/ha}$ after removing sub-hectare rounding noise (`< 1.0 ha`).
2. **PSA OpenSTAT Provincial Farmgate Prices (`DB/2M/NFG/0032M4AFN01..08.px`, 2010–2025):** **18,967 annual provincial farmgate prices** (`data/psa_provincial_farmgate_prices.csv`).
3. **PSA Table 5.5/5.6/5.7 Palay Costs & Returns (`DB/3S/C5`, 2022–2024):** **216 rows** (`2024 All Palay Average Total Costs = ₱59,695/ha`). Non-palay crops use transparent, user-editable reference cost estimates.
4. **DOST-PAGASA CliMap v2.1 Normals (1991–2020):** Observed monthly/annual GeoTIFF rainfall & mean temperature normals across all 82 provinces (`data/pagasa_climap_province_normals.csv` & `philippine_provinces_crop_suitability.*`).
5. **NASA POWER Agroclimatology MERRA-2 (1987–2025):** **3,198 province-year records** (`r = 0.873` rain and `r = 0.725` temp spatial correlation vs. DOST-PAGASA CliMap normals).

*(For complete dataset provenance, unit conversions, manual PhilRice/PAGASA import hooks, and per-crop metrics, see [`PHILIPPINE_DATASET_DOCUMENTATION.md`](PHILIPPINE_DATASET_DOCUMENTATION.md).)*

---

## 2. Machine Learning Architecture & 5-Model Evaluation (`train_model.py`)

* **ML Panel:** Uniform **2010–2025 multi-crop panel (`32,618` rows)** split chronologically into **Train/Validation (`2010–2021`, 24,369 rows)** vs. **Unseen Out-of-Time Holdout Test (`2022–2025`, 8,249 rows)** with zero key leakage.
* **Verified Inference Features (4 Only):** `item` (Crop Type), `area` (Philippine Province), `average_rain_fall_mm_per_year`, `avg_temp`.
* **Automatic Model Selection (5 Core CSST 102 Algorithms):** Strictly selected via **5-Fold Cross-Validation $R^2$ inside `2010–2021`** (`Linear Regression`, `Ridge Regression`, `Decision Tree Regressor`, `Random Forest Regressor`, and `Gradient Boosting Regressor`):
  * **Winner:** **Random Forest Regressor** — 5-Fold CV $R^2 = \mathbf{0.9246}$ ($\pm 0.0075$), Out-of-Time Test $R^2 = \mathbf{0.7332}$ (Seen Crops $R^2 = 0.7369$), Test MAE $= 20{,}447.81\text{ hg/ha}$ ($\approx 2.04\text{ MT/ha}$).
  * *Transparent Per-Crop Routing:* **28 well-sampled crops** (`Rice, paddy` $R^2=0.824$, `Maize` $R^2=0.711$, `Cassava`, `Tomato`, `Sugarcane`, `Banana Cavendish`, etc.) run on **`philippine_ml`**, while **7 sparse/low-predictability or newly disaggregated crops** (`Potatoes`, `Red Onion`, `Yellow Onion`, `Garlic`, `Peanut`, `Sorghum`, `Soybeans`) use verified **`psa_statistical_historical`** yields cleanly.

*(Note: $R^2$ is the Coefficient of Determination measuring variance explained across crops & provinces, not literal accuracy percentage.)*

---

## 3. Running & Testing Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Refresh datasets / retrain ML & regenerate static charts
python build_philippines_real_dataset.py
python train_model.py
python generate_charts.py

# 3. Start web application
python app.py   # Open http://127.0.0.1:5000/

# 4. Run full verification suites
python test_app_in_memory.py
python test_new_api_routes.py
python test_e2e_full_system.py
python test_philippine_upgrade.py
```

---
*CSST 102 – Basic Machine Learning Project, Laguna State Polytechnic University.*
