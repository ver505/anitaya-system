# AniTaya – Philippine Crop Recommendation, Harvest Prediction & Income Estimation System
**Web-Based Predictive Machine Learning Decision-Support System for Filipino Farmers**

**Institution:** Laguna State Polytechnic University (LSPU) – College of Computer Studies  
**Course:** CSST 102 – Basic Machine Learning | 1st Semester AY 2026–2027  

---

## 1. System Overview & Authoritative Philippine Datasets

**AniTaya** (*Ani* = Harvest, *Taya* = Forecast/Estimate) forecasts Philippine crop harvest yields and net farm income across **all 82 Philippine provinces** and **35 crops** using verified Philippine public data sources (license/public access permits full redistribution in `data/`):

1. **PSA OpenSTAT Crop Production & Harvested Area (`DB/2E/CS`, 1987–2025):**  
   - **URL:** `https://openstat.psa.gov.ph/PXWeb/api/v1/en/DB/2E/CS/` (`0012E4EVCP0.px`, `0022E4EAHC0.px`, `0082E4EVCP3.px`, `0112E4EAHM3.px`, `0072E4EVCP2.px`, `0102E4EAHM2.px`, `0072E4EVCP1.px`, `0092E4EAHM1.px`)  
   - **Saved File:** `data/psa_provincial_production.csv` (**36,088 clean records**, 82 provinces × 35 crops).
2. **PSA OpenSTAT Provincial Farmgate Prices (`DB/2M/NFG`, 2010–2025):**  
   - **URL:** `https://openstat.psa.gov.ph/PXWeb/api/v1/en/DB/2M/NFG/0032M4AFN01.px` .. `0032M4AFN08.px`  
   - **Saved File:** `data/psa_provincial_farmgate_prices.csv` (**18,967 annual records**).
3. **PSA Table 5.5/5.6/5.7 Palay Costs & Returns (`DB/3S/C5`, 2022–2024):**  
   - **URL:** `https://openstat.psa.gov.ph/PXWeb/api/v1/en/DB/3S/C5/0052E2BARP0.px` .. `0072E2BARN0.px`  
   - **Saved File:** `data/psa_palay_costs_returns.csv` (**216 rows**; 2024 All Palay Average Total Costs = `₱59,695/ha`). Other crops use editable reference estimates.
4. **DOST-PAGASA CliMap v2.1 Gridded Climate Normals (1991–2020):**  
   - **URL:** `https://climgridph.pagasa.dost.gov.ph/climapv2/` (`PH_Adm2_Provinces.json`, `maps/rr/tif/01..12.tif`, `maps/tm/tif/01..12.tif`)  
   - **Saved Files:** `data/pagasa_climap_province_normals.csv` (82 provinces) and synchronized climate values in `data/philippine_provinces_crop_suitability.csv` & `.json`.
5. **NASA POWER Agroclimatology MERRA-2 Time Series (1987–2025):**  
   - **URL:** `https://power.larc.nasa.gov/api/temporal/monthly/point` (`community=AG`, `T2M`, `PRECTOTCORR`)  
   - **Saved Files:** `data/philippines_province_yearly_climate.csv` (**3,198 province-year rows**; cross-checked vs. PAGASA normals: $r=0.873$ rainfall, $r=0.725$ temperature).
6. **Master Merged Dataset:** `data/philippines_crop_yield_climate.csv` (**36,088 clean deduplicated rows**, zero nulls/duplicates).

*(Full URLs, cleaning formulas, manual PhilRice PalayStat drop-in instructions, and per-crop audit are documented in [`PHILIPPINE_DATASET_DOCUMENTATION.md`](PHILIPPINE_DATASET_DOCUMENTATION.md).)*

---

## 2. Machine Learning Workflow & 5-Model Comparison (`train_model.py` & `notebooks/`)

* **Training Panel:** Uniform **2010–2025 multi-crop panel (`32,618` rows)** split chronologically into **Train/Validation (`2010–2021`, 24,369 rows)** vs. **Unseen Out-of-Time Holdout Test (`2022–2025`, 8,249 rows)** with zero key leakage.
* **Verified Inference Features (4 Inputs Only):** `item` (Crop Type), `area` (Philippine Province), `average_rain_fall_mm_per_year`, and `avg_temp` (`OneHotEncoder` + `StandardScaler` inside Scikit-Learn `Pipeline`).
* **CSST 102 Model Selection (Strictly by 5-Fold Validation CV $R^2$ inside `2010–2021`):**
  * **Winner:** **Random Forest Regressor** — 5-Fold CV $R^2 = \mathbf{0.9246}$ ($\pm 0.0075$), Holdout Test $R^2 = \mathbf{0.7332}$ (Seen Crops $R^2 = 0.7369$), Test MAE $= 20{,}447.81\text{ hg/ha}$ ($\approx 2.04\text{ MT/ha}$), RMSE $= 50{,}030.50\text{ hg/ha}$.
  * Compared against: `Gradient Boosting Regressor` (`CV R²=0.8202`, `Test R²=0.6577`), `Decision Tree Regressor` (`CV R²=0.8083`, `Test R²=0.6228`), `Ridge Regression` (`CV R²=0.5093`, `Test R²=0.3961`), and `Linear Regression (Baseline)` (`CV R²=0.5093`, `Test R²=0.3958`).
  * **Per-Crop Routing:** **28 well-sampled crops** (`Rice, paddy` $R^2=0.824$, `Maize` $R^2=0.711$, `Cassava`, `Tomato`, `Sugarcane`, etc.) use **`philippine_ml`**; **7 sparse or newly disaggregated crops** (`Potatoes`, `Red Onion`, `Yellow Onion`, `Garlic`, `Peanut`, `Sorghum`, `Soybeans`) use verified **`psa_statistical_historical`** yields cleanly.
  * *(Note: $R^2$ is Coefficient of Determination / variance explained, not a literal percentage accuracy guarantee.)*

---

## 3. Files That Must Be Generated Before Running the App

All required generated files are **already pre-generated and included in this repository/ZIP** so you can launch the web app immediately. If regenerating cleanly from scratch, execute these scripts in exact order before starting `app.py`:

| Step | Script / Notebook | Files Generated Before Running App |
| :---: | :--- | :--- |
| 1 | `python build_philippines_real_dataset.py` *(requires internet for PSA/PAGASA/NASA POWER APIs)* | `data/pagasa_climap_province_normals.csv`, `data/philippines_province_yearly_climate.csv`, `data/psa_provincial_production.csv`, `data/psa_provincial_farmgate_prices.csv`, `data/psa_palay_costs_returns.csv`, `data/philippines_crop_yield_climate.csv`, synced `data/philippine_provinces_crop_suitability.{csv,json}` |
| 2 | `python train_model.py` *(also mirrored step-by-step in `notebooks/anitaya_eda_and_model_training.ipynb`)* | `models/anitaya_best_pipeline.joblib` (`36.86 MB` trained Random Forest pipeline) & `models/model_metrics.json` (benchmark metrics, feature importances, holdout samples, benchmarks, per-crop routing table) |
| 3 | `python generate_charts.py` | `static/images/model_comparison.png`, `static/images/feature_importance.png`, `static/images/actual_vs_predicted.png` |
| 4 | Runtime (`python app.py`) | Auto-creates `data/taya_history.db` on first startup via `history_db.py` (SQLite prediction log) |

---

## 4. Installation, Running & Automated Testing Instructions

### Prerequisites
- Python **3.10 – 3.13** (`3.13.0` specified in `.python-version`)

### Install & Run Locally
```bash
# 1. Extract ZIP / open project folder
cd anitaya-system

# 2. (Optional) Create & activate virtual environment, then install dependencies
pip install -r requirements.txt

# 3. Start the Flask Web Application (or double-click run_website.bat on Windows)
python app.py
```
Open your browser at: **`http://127.0.0.1:5000/`**

### Run Verification Suites
```bash
python test_app_in_memory.py
python test_new_api_routes.py
python test_e2e_full_system.py
python test_philippine_upgrade.py
```

---
*CSST 102 – Basic Machine Learning Project, Laguna State Polytechnic University.*
