# AniTaya Philippine Dataset & Machine Learning Documentation

This document records the complete data engineering, source verification, cleaning methodology, unit normalization, machine learning evaluation protocol, per-crop prediction routing, and limitations of the upgraded **AniTaya** Philippine Crop Recommendation, Harvest Prediction, and Income Estimation System.

---

## 1. Authoritative Data Sources & URLs Used

All datasets integrating Philippine crop production, farmgate prices, cost benchmarks, and agro-climate series are fetched via `build_philippines_real_dataset.py` from public official endpoints (with zero synthetic formulas and zero Indonesia proxies):

### A. Philippine Crop Production & Harvested Area (Yield Denominator)
* **Source:** Philippine Statistics Authority (PSA) OpenSTAT PxWeb REST API (`DB/2E/CS`)
  * **Base API URL:** `https://openstat.psa.gov.ph/PXWeb/api/v1/en/DB/2E/CS/`
  * **Palay & Corn (1987–2025, Annual by Province):**
    * Production (Metric Tons): `0012E4EVCP0.px`
    * Harvested Area (Hectares): `0022E4EAHC0.px`
    * Valid matched observations retained: **6,044 rows**
  * **Vegetables & Root Crops (2010–2025, Annual by Province):**
    * Production (MT): `0082E4EVCP3.px`
    * Area Harvested (HA): `0112E4EAHM3.px`
    * Valid matched observations retained: **16,198 rows**
  * **Fruit Crops (2010–2025, Annual by Province):**
    * Production (MT): `0072E4EVCP2.px`
    * Area Harvested (HA): `0102E4EAHM2.px`
    * Valid matched observations retained: **7,869 rows**
  * **Non-Food & Industrial Crops (2010–2025, Annual by Province):**
    * Production (MT): `0072E4EVCP1.px`
    * Area Harvested (HA): `0092E4EAHM1.px`
    * Valid matched observations retained: **5,977 rows**
* **Output File:** `data/psa_provincial_production.csv` (**36,088 clean rows** across **82 provinces** and **35 crops**).
* *Note on Web Portal vs. PxWeb API:* Interactive web URLs (`PXWeb/pxweb/en/...` and `Metadata/2E4EAHM0`) sit behind browser challenge protection; therefore AniTaya uses PSA OpenSTAT's official public `PXWeb/api/v1/en/` JSON/CSV REST API directly.

### B. Philippine Farmgate Prices & Palay Costs/Returns
* **PSA Official Farmgate Prices (New Series, 2010–2025 Annual by Province):**
  * **URLs:** `https://openstat.psa.gov.ph/PXWeb/api/v1/en/DB/2M/NFG/0032M4AFN01.px` through `0032M4AFN08.px`
  * **Output File:** `data/psa_provincial_farmgate_prices.csv` (**18,967 provincial annual price records** across all 8 tables).
  * **Unit Notes:**
    * Most crops are officially reported in `PHP/kg`.
    * `Coconut matured` (`AFN08`) is officially reported in `PHP/nut (PSA Official)` (`~₱7.57/nut` median; roughly comparable on a ₱/kg-equivalent basis given ~1.2 kg average weight per whole coconut with husk, labeled transparently in API/UI as *PHP/kg-equiv (from PHP/nut PSA)*).
    * `Sugarcane` production (`0072E4EVCP1.px`) measures whole millable cane (`MT`), whereas PSA farmgate `AFN08` tracks retail `Sugarcane for Chewing` (~₱11.92/kg). Using chewing-cane retail price on 60+ MT/ha commercial centrifugal cane would overstate mill-gate revenue; therefore `crop_engine.py` uses a commercial mill-gate cane reference of **₱2.80/kg (`₱2,800/MT`, fully editable in UI)** and documents that clearly.
* **PSA Table 5.5 / 5.6 / 5.7 Costs and Returns of Palay Production (2022–2024):**
  * **URLs:** `DB/3S/C5/0052E2BARP0.px` (All Palay), `0062E2BARI0.px` (Irrigated), `0072E2BARN0.px` (Non-Irrigated)
  * **Output File:** `data/psa_palay_costs_returns.csv` (**216 rows**).
  * Used directly for `Rice, paddy` default production cost (`2024 All Palay Average Total Costs = ₱59,695.00/ha`). For non-palay crops where no PSA annual provincial cost survey is published via API, AniTaya uses documented **Reference Production Cost Estimates** (labeled `"is_official_psa_cost": false` and editable directly in the planner).

### C. Philippine Gridded Climate Normals & Historical Yearly Weather
* **DOST-PAGASA CliMap v2.1 / ClimGridPh Observed Normals (1991–2020):**
  * **URL:** `https://climgridph.pagasa.dost.gov.ph/climapv2/`
  * **Files Extracted:** `assets/js/PH_Adm2_Provinces.json` + observed monthly GeoTIFFs `maps/rr/tif/01.tif`..`12.tif` (rainfall mm) and `maps/tm/tif/01.tif`..`12.tif` (mean temperature °C).
  * **Output File:** `data/pagasa_climap_province_normals.csv` (**82 provinces**), also synchronized directly into `data/philippine_provinces_crop_suitability.csv` and `.json`.
* **NASA POWER Agroclimatology Monthly API (MERRA-2 Reanalysis, 1987–2025):**
  * **URL:** `https://power.larc.nasa.gov/api/temporal/monthly/point` (`community=AG`, `T2M` in °C, `PRECTOTCORR` in mm/day)
  * **Output File:** `data/philippines_province_yearly_climate.csv` (**3,198 province-year records**, matched uniformly to every province polygon centroid from 1987–2025).
  * **Empirical Climate Cross-Check (1991–2020 NASA POWER vs. DOST-PAGASA CliMap v2.1 GeoTIFFs across 82 provinces):**
    * Annual Rainfall spatial correlation: **$r = 0.873$**
    * Mean Temperature spatial correlation: **$r = 0.725$**
  * *(Note: Open-Meteo ERA5 Reanalysis hit an HTTP 429 daily quota block after 40 provinces and was replaced uniformly by NASA POWER MERRA-2 across all 82 provinces so training uses one consistent meteorological product.)*

### D. Endpoints Requiring Optional Manual CSV Drop-in (`data/manual_imports/`)
During automated HTTP probing, two external government web portals timed out (`>20–35s` server timeout):
1. **PhilRice PalayStat Table 1:** `https://palaystat.philrice.gov.ph/statistics/retrieve/table/1`
2. **PAGASA ClimGridPh Station History PHP APIs:** `APIgetHSProvince.php` and `MunCodeAPI.php`

**Manual Download Hook:** Users with browser access or official statistical releases can save CSV exports into `data/manual_imports/` containing columns `area` (or `province`), `year`, `item`, and `value_hg_ha` (or `yield_mt_ha` or `production_mt` + `harvested_area_ha`). Re-running `python build_philippines_real_dataset.py` automatically merges those manual files into `philippines_crop_yield_climate.csv`.

---

## 2. Cleaning, Deduplication, Merging & Unit Conversion Rules

1. **Province Name Reconciliation:** Mapped historical PSA province renames (e.g., `Compostela Valley` -> `Davao de Oro`, `North Cotabato` -> `Cotabato`, `Western Samar` -> `Samar`, `Negros Occidental a/` -> `Negros Occidental`), summed `Maguindanao del Norte + Maguindanao del Sur` back to canonical `Maguindanao` where split, and aligned Metro Manila (`NCR`).
2. **Exact Yield Calculation:**
   $$\text{yield\_mt\_ha} = \frac{\text{production\_mt}}{\text{harvested\_area\_ha}}, \qquad \text{value\_hg\_ha} = \text{yield\_mt\_ha} \times 10{,}000$$
3. **Artifact & Outlier Filtering:**
   * Excluded records with sub-hectare reported harvested area (`harvested_area_ha < 1.0 ha`), eliminating 573 small-denominator rounding spikes (such as a `0.04 ha` soybean entry producing an implausible `135 MT/ha`).
   * Excluded records outside `[0.02, 200.0] MT/ha` physical bounds.
   * Dropped rows missing target or key inputs (`0` nulls remaining) and deduplicated strictly on `(area, year, item)` (`0` duplicate keys remaining).
4. **Master Philippine Dataset Output:**
   * **File:** `data/philippines_crop_yield_climate.csv`
   * **Dimensions:** **36,088 clean records** × 20 columns (**82 provinces**, **35 crops**, **1987–2025**; legacy global dataset preserved untouched in `data/df_yield_merged.csv` + `backups/pre_ph_upgrade/`).

---

## 3. Feature Selection & Leakage Prevention

1. **Removed Unreliable Features:**
   * Permanently removed `pesticides_tonnes` (which previously defaulted to a hardcoded `120.0`) and `area="Indonesia"` (which mapped Philippine predictions onto Indonesian country rows).
   * Explicitly excluded post-harvest variables (`production_mt`, `harvested_area_ha`, `yield_mt_ha`) and target `value_hg_ha` from training inputs.
2. **Verified Inference-Identical Feature Set (4 Inputs):**
   * `item` — Crop name (`OneHotEncoder(handle_unknown="ignore")`)
   * `area` — Philippine province (`OneHotEncoder(handle_unknown="ignore")`)
   * `average_rain_fall_mm_per_year` — Year-matched annual precipitation in mm (`StandardScaler`)
   * `avg_temp` — Year-matched mean temperature in °C (`StandardScaler`)
3. **Informational Soil Type Policy:**
   * `soil_type` (DA-BSWM dominant provincial soil series) is surfaced in the UI strictly as **agronomic advisory metadata (`soil_is_ml_feature = False`)** and is never falsely represented as an ML input.
4. **Leak-Free Pipeline & Homogeneous Modeling Era (`2010–2025`):**
   * Because only Palay and Corn exist in PSA tables from `1987–2009` while the other 33 crops begin at `2010`, ML training (`train_model.py`) uses the uniform **2010–2025 multi-crop panel (`32,618` rows)** while preserving the full **36,088-row (1987–2025)** archive in `dataset_info` and PSA lookups.
   * **Chronological Holdout Split:** Train/Validation **`2010–2021` (`24,369` rows)** vs. Unseen Future Holdout Test **`2022–2025` (`8,249` rows; `8,194` in training-seen crops)**.
   * Inside `2010–2021`, all preprocessing transformers fit independently inside each **5-Fold Cross-Validation** fold (`KFold(5, shuffle=True, random_state=42)`), alongside an explicit **5-Fold Province `GroupKFold`** generalization diagnostic.
   * Model selection depends **strictly on Validation 5-Fold CV $R^2$** (`CV_R2_Mean`, tie-broken by `-CV_RMSE`) with zero peeking at the holdout test split.

---

## 4. Model Comparison Results (`models/model_metrics.json`)

Following the CSST 102 guideline priority (*explainable ML workflow over unnecessary model complexity*), 5 core regression algorithms were evaluated cleanly on the `2010–2021` train / `2022–2025` holdout protocol (`Random Forest Regressor` selected by Validation CV $R^2$, then refitted on the complete `2010–2025` panel into `36.86 MB` `models/anitaya_best_pipeline.joblib` using `compress=3`):

| Rank (by CV $R^2$) | Algorithm | 5-Fold CV $R^2$ | Holdout Test $R^2$ (`2022–2025`) | Seen-Crops Test $R^2$ | Test MAE (hg/ha) | Test RMSE (hg/ha) | Province GroupKFold $R^2$ | Fit Time | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Random Forest Regressor** | **0.9246 ± 0.0075** | **0.7332** | **0.7369** | **20,447.81** (`2.04 MT/ha`) | **50,030.50** | 0.2855 | 36.42s | **SELECTED WINNER** |
| 2 | Gradient Boosting Regressor | 0.8202 ± 0.0064 | 0.6577 | 0.6610 | 30,083.31 | 56,662.94 | 0.4299 | 48.28s | Evaluated |
| 3 | Decision Tree Regressor | 0.8083 ± 0.0302 | 0.6228 | 0.6256 | 29,671.75 | 59,485.86 | 0.3086 | 0.87s | Evaluated |
| 4 | Ridge Regression | 0.5093 ± 0.0125 | 0.3961 | 0.3974 | 41,611.77 | 75,263.34 | 0.4398 | 0.10s | Evaluated |
| 5 | Linear Regression (Baseline) | 0.5093 ± 0.0126 | 0.3958 | 0.3971 | 41,669.83 | 75,282.38 | 0.4375 | 0.37s | Baseline |

* **Feature Importance (Impurity):** `Crop Type (item)` **49.46%**, `Philippine Province (area)` **40.76%**, `Average Temperature (avg_temp)` **7.94%**, `Annual Rainfall (average_rain_fall_mm_per_year)` **1.84%**.
* **Permutation Importance (Out-of-Time Test $R^2$ Drop):** `Crop Type (item)` **65.56%**, `Philippine Province (area)` **28.68%**, `Average Temperature` **5.76%**, `Annual Rainfall` **0.00%**.
* *Interpretation Note:* Overall Test $R^2$ (`0.7332`) is the Coefficient of Determination measuring variance explained across diverse crops and provinces—**not** a literal percentage accuracy claim. Notice also why `GroupKFold_Province_R2` (`0.2855` for Random Forest vs `0.4398` Ridge) drops compared to temporal test: when a whole province's baseline historical productivity is completely withheld during spatial cross-validation, annual climate + crop identity alone cannot infer unobserved local irrigation infrastructure, soil fertility, or farm scale.

---

## 5. Transparent Per-Crop Prediction Routing (`28 ML` vs. `7 Historical PSA`)

Instead of forcing low-predictability or newly disaggregated crops through cross-crop ML extrapolation, `crop_engine.py` classifies all 35 crops automatically from `models/model_metrics.json["per_crop_metrics"]`:

* **Tier 1 (`philippine_ml` — 28 crops):** Used when `train_rows >= 250`, `provinces >= 15`, and unseen `2022–2025` Holdout `test_r2 >= 0.10`. Key holdout results:
  * `Plantains and others` ($R^2 = 0.8304$), `Rice, paddy` ($R^2 = 0.8236$, $\text{MAE} = 0.272\text{ MT/ha}$), `Rubber` ($R^2 = 0.7758$), `Coconut` ($R^2 = 0.7649$), `Pineapple` ($R^2 = 0.7306$), `Mango` ($R^2 = 0.7286$), `Maize` ($R^2 = 0.7109$, $\text{MAE} = 0.543\text{ MT/ha}$), `Calamansi` ($R^2 = 0.6687$), `Tomato` ($R^2 = 0.6479$), `Abaca` ($R^2 = 0.6337$), `Papaya` ($R^2 = 0.6270$), `Banana Cavendish` ($R^2 = 0.5995$), `Cassava` ($R^2 = 0.5869$), `Squash` ($R^2 = 0.5618$), `Sugarcane` ($R^2 = 0.5461$), `Cabbage` ($R^2 = 0.5430$), `Sweet potatoes` ($R^2 = 0.5100$), `Yams` ($R^2 = 0.4806$), `Taro` ($R^2 = 0.4775$), `String Beans` ($R^2 = 0.4770$), `Watermelon` ($R^2 = 0.4761$), `Eggplant` ($R^2 = 0.4537$), `Tobacco` ($R^2 = 0.4521$), `Coffee` ($R^2 = 0.4414$), `Mungbean` ($R^2 = 0.2505$), `Pechay` ($R^2 = 0.2405$), `Bitter Gourd` ($R^2 = 0.2205$), `Cacao` ($R^2 = 0.1678$).
* **Tier 2 (`psa_statistical_historical` — 7 crops):**
  * `Potatoes` (140 train rows, concentrated highlands), `Garlic` (206 train rows), `Sorghum` (107 train rows), `Soybeans` (188 train rows, Test $R^2 = 0.0567$), `Peanut` (low out-of-time predictability, Test $R^2 = 0.0533$), `Red Onion` (0 pre-2022 disaggregated PSA rows; starts 2022), `Yellow Onion` (0 pre-2022 disaggregated PSA rows; starts 2022).
  * For these 7 crops, AniTaya returns the verified recent **PSA provincial historical median yield** (or national producing-province median clearly labeled as national fallback if that province has no PSA commercial record for that crop).
* **Tier 3 (`fallback_reference`):** Triggered only if an unrecognized/custom crop lacks both ML weights and PSA history, labeled explicitly as a reference baseline estimate.

---

## 6. Known Remaining Limitations

1. **Annual Climate Aggregation:** Year-level rainfall/temperature means do not capture extreme sub-weekly typhoon wind speeds, flash flood timing during flowering, or localized pest outbreaks.
2. **Non-Palay Production Costs:** Official PSA Table 5.5/5.6/5.7 cost-and-return surveys cover Palay cleanly; non-palay costs rely on category/crop reference defaults until users enter their own ₱/ha budget in the planner.
3. **Non-ML Advisory Signals:** DA-BSWM dominant soil series and 12-month agro-calendars are rule-based advisories and do not enter the trained ML pipeline.
4. **Zero Financial Guarantee:** All yield forecasts and peso income projections remain planning estimates only.
