# AniTaya – Agricultural Harvest and Income Prediction System
**Web-Based Predictive Machine Learning System for Filipino Farmers**

**Institution:** Laguna State Polytechnic University (LSPU) – College of Computer Studies  
**Course:** CSST 102 – Basic Machine Learning | 1st Semester AY 2026–2027  
**Faculty / Instructor:** Anthony U. Dorado  
**Program Coordinator:** Al John A. Villareal  
**Associate Dean:** Ronnel A. Dela Cruz  

---

## 1. Project Overview & Problem Formulation

### 1.1 The Practical Problem
In the Philippines, smallholder farmers often make planting and resource-allocation decisions based on intuition or historical weather patterns that are increasingly disrupted by climate variability. Crucial decisions—such as crop selection, input purchasing (fertilizers and agrochemicals), and credit/financing—are made without reliable data on expected harvest volume and financial return.

### 1.2 System Purpose
**AniTaya** (*Ani* = Harvest, *Taya* = Forecast/Estimate) is an end-to-end web-based predictive machine learning application designed to forecast:
1. **Expected Harvest Yield**: in Metric Tons (MT), Kilograms (kg), and standard 50-kg bags/cavans (*kaban ng palay*).
2. **Projected Farmer Income (in Philippine Peso ₱ - PHP)**: Estimated gross revenue, complete production cost breakdown (seeds, fertilizer, labor, fuel/machinery), net farmer profit, and Return on Investment (ROI %).
3. **Climate Sensitivity & Risk Scenarios**: Favorable season (+15%), Baseline expected yield, and Adverse weather / typhoon shock (-20%).

---

## 2. Dataset Description & Citation

* **Dataset Title:** Dataset for Crop Yield Prediction
* **Source:** Zenodo (compiled from Food and Agriculture Organization - FAOSTAT and climate records)
* **Zenodo Record ID:** `18761663`
* **DOI:** `10.5281/zenodo.18761663`
* **URL:** [https://zenodo.org/records/18761663](https://zenodo.org/records/18761663)
* **Master Merged Dataset:** data/df_yield_merged.csv (25,932 records x 7 attributes)
* **Missing Values:** 0 missing entries across all columns (100% clean and validated)

### 2.1 Uncombined Raw Source Datasets (data/)
In addition to the cleaned and merged master dataset (df_yield_merged.csv), the repository stores the 4 original uncombined source CSV datasets, preserving full data provenance:
1. **data/yield.csv** (4.04 MB - 56,717 records): Historical crop production yields (Value in hg/ha) per Country (Area), Crop (Item), and Year from FAOSTAT.
2. **data/rainfall.csv** (136 KB - 6,727 records): Historical annual precipitation (verage_rain_fall_mm_per_year) per Country and Year.
3. **data/pesticides.csv** (447 KB - 4,349 records): Annual agricultural agrochemical consumption (Value in tonnes of active ingredients) from FAOSTAT Agri-Environmental indicators.
4. **data/temp.csv** (1.42 MB - 71,311 records): Annual mean land surface temperatures (vg_temp in deg C) per Country and Year.
5. **data/philippine_provinces_crop_suitability.csv & .json**: Official 82-province agricultural recommendation and agro-climatic profile dataset based on DA-AMIA (NCCAG v2.0), PSA, DA-BSWM, and PAGASA.

### 2.2 Merged Dataset Attributes:
| Attribute | Type | Description | Unit / Range |
| :--- | :--- | :--- | :--- |
| `item` | Categorical | Crop type (Rice paddy, Maize, Cassava, Potatoes, Sweet potatoes, Plantains, etc.) | 10 agricultural crops |
| `average_rain_fall_mm_per_year` | Numerical | Annual precipitation | 51.0 to 3,240.0 mm |
| `pesticides_tonnes` | Numerical | Agrochemical / crop care input | 0.04 to 367,778.0 tonnes |
| `avg_temp` | Numerical | Average regional temperature | 1.3°C to 30.65°C |
| `area` | Categorical | Country or agro-ecological zone | 101 countries |
| `year` | Integer | Recorded agricultural observation year | 1990 to 2013 |
| **`value_hg_ha`** *(Target)* | Integer | **Crop harvest yield rate** | **50 to 501,412 hg/ha** |

---

## 3. Machine Learning Methodology & Workflow

### 3.1 Preprocessing & Data Leakage Prevention
To guarantee scientific validity and prevent data leakage:
1. **Train/Test Splitting:** An **80/20 train-test split** (`random_state=42`) was performed before any transformation.
2. **ColumnTransformer Pipeline:**
   - **Categorical Features (`item`, `area`):** Encoded using `OneHotEncoder(handle_unknown='ignore')`.
   - **Numerical Features (`average_rain_fall_mm_per_year`, `pesticides_tonnes`, `avg_temp`):** Normalized using `StandardScaler()`.
3. **Unified Pipeline:** All transformations and estimators are bundled inside a Scikit-Learn `Pipeline` saved as `anitaya_best_pipeline.joblib`. Submitted web inputs pass through the exact same transformation steps.

### 3.2 Model Comparison (CSST 102 Development Requirement)
We compared five regression architectures across identical splits:

| Model Architecture | MAE (hg/ha) | RMSE (hg/ha) | Test $R^2$ Score | 3-Fold CV $R^2$ | Training Time | Selection Justification |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Linear Regression (Baseline)** | 30,572.39 | 43,933.90 | 0.7405 | 0.7378 (±0.010) | 1.0 s | Simple linear baseline; cannot capture non-linear climate thresholds |
| **Ridge Regression** | 30,537.31 | 43,949.49 | 0.7403 | 0.7375 (±0.009) | 0.3 s | L2 regularized linear model; near-identical performance to OLS |
| **Decision Tree Regressor** | 9,225.55 | 19,688.43 | 0.9479 | 0.9170 (±0.000) | 0.8 s | Captures non-linear thresholds but prone to regional variance |
| **Random Forest (Primary Model)** | **6,908.50** | **15,903.82** | **0.9660** | **0.9395 (±0.007)** | **43.4 s** | **WINNER (Selected): Highest $R^2$ (96.60%), lowest error, ensemble robustness** |
| **Gradient Boosting Regressor** | 14,927.96 | 23,675.23 | 0.9246 | 0.9191 (±0.009) | 32.9 s | Solid ensemble boosting, but higher MAE than Random Forest |

### 3.3 Feature Importance Analysis
From the trained Random Forest regressor, feature importance was quantified:
1. **Crop Type (`item`):** 60.46% (Primary driver of baseline yield potential)
2. **Country / Regional Soil Context (`area`):** 20.64%
3. **Agrochemical / Input Level (`pesticides_tonnes`):** 10.02%
4. **Average Temperature (`avg_temp`):** 4.91%
5. **Precipitation (`average_rain_fall_mm_per_year`):** 3.97%

---

## 4. Philippine Agricultural Integration & Financial Modeling

AniTaya maps FAO yield rates directly to Philippine Department of Agriculture (DA) and Philippine Statistics Authority (PSA) farmgate benchmarks:

* **Palay / Rice paddy (`Rice, paddy`):** Farmgate ₱22.00/kg; production cost ₱45,000/ha; 50 kg/cavan.
* **Mais / Corn (`Maize`):** Farmgate ₱20.50/kg; production cost ₱38,000/ha.
* **Kamoteng Kahoy (`Cassava`):** Farmgate ₱14.00/kg; production cost ₱32,000/ha.
* **Patatas (`Potatoes`):** Farmgate ₱45.00/kg; production cost ₱95,000/ha (highland seed tubers).
* **Kamote (`Sweet potatoes`):** Farmgate ₱28.00/kg; production cost ₱28,000/ha.
* **Saging / Saba (`Plantains and others`):** Farmgate ₱25.00/kg; production cost ₱40,000/ha.

### Financial Formulas:
$$\text{Yield Rate (MT/ha)} = \frac{\text{Predicted } hg/ha}{10,000}$$
$$\text{Total Harvest (kg)} = \left(\frac{\text{Predicted } hg/ha}{10}\right) \times \text{Land Area (ha)}$$
$$\text{Gross Revenue (₱)} = \text{Total Harvest (kg)} \times \text{Farmgate Price (₱/kg)}$$
$$\text{Net Farmer Profit (₱)} = \text{Gross Revenue (₱)} - \left(\text{Cost per ha (₱)} \times \text{Land Area (ha)}\right)$$
$$\text{ROI (\%)} = \left(\frac{\text{Net Profit}}{\text{Total Production Cost}}\right) \times 100$$

---

## 5. Web Application Architecture

* **Backend:** Python + Flask (`app.py`)
* **ML Inference:** Scikit-Learn Pipeline (`anitaya_best_pipeline.joblib`)
* **Frontend:** Responsive HTML5 + Tailwind CSS + Chart.js
* **Validation:** Server-side range checks (`land_area > 0`, `rainfall: 100-6000 mm`, `temp: 10-48°C`) returning structured HTTP 422 errors.
* **Interactive Demo Presets:** 1-Click buttons to instantly demonstrate real-world scenarios (Laguna Rice, Isabela Corn, Benguet Potato, Quezon Cassava).

---

## 6. Responsible Machine Learning & Limitations (Section 4 Rubric)

1. **Advisory Decision Support:** Predictions are statistical estimates derived from historical agronomic trends. They serve as an agricultural advisory guide and do not constitute guaranteed financial returns.
2. **Severe Climate Hazards:** The model assumes standard seasonal weather. Catastrophic weather shocks (Category 5 typhoons, flash floods, El Niño droughts) require localized disaster mitigation and crop insurance.
3. **Data Privacy:** AniTaya collects zero farmer personally identifiable information (PII) or land title documents. All calculations occur statelessly.
4. **Transparency & Fairness:** Feature importances, evaluation scores, and model baselines are openly displayed to prevent black-box opacity.

---

## 7. How to Run Locally (Instructor Evaluation Instructions)

### Prerequisites
* Python 3.10, 3.11, 3.12, or 3.13 installed.

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2 (Optional): Re-train Models & Generate Visuals
To verify reproducibility from scratch:
```bash
python train_model.py
python generate_charts.py
```

### Step 3: Run the Web Application
```bash
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000/
```

### Step 4: Run Automated In-Memory Tests
```bash
python test_app_in_memory.py
```

---

## 8. Sample Test Cases for Demonstration

| Test Case | Crop | Land Area | Season | Location | Predicted Yield | Estimated Net Profit |
| :--- | :--- | :---: | :--- | :--- | :---: | :---: |
| **Case 1: Laguna Rice Farmer** | Palay (`Rice, paddy`) | 2.0 ha | Wet Season | Laguna / CALABARZON | ~6.42 MT (128 cavans) | **~₱51,180 (ROI: 56.8%)** |
| **Case 2: Isabela Corn Farmer** | Mais (`Maize`) | 3.5 ha | Dry Season | Cagayan Valley | ~14.8 MT (296 cavans) | **~₱170,400 (ROI: 128.1%)** |
| **Case 3: Benguet Potato Farmer**| Patatas (`Potatoes`) | 0.8 ha | Cool Dry | Benguet Highlands | ~14.1 MT (282 cavans) | **~₱558,500 (ROI: 734.9%)** |
| **Case 4: Quezon Cassava Grower**| Kamote (`Cassava`) | 1.5 ha | Wet Season | CALABARZON | ~18.3 MT (366 cavans) | **~₱208,200 (ROI: 433.8%)** |

---
*Created for CSST 102 – Basic Machine Learning Final Project Requirement, Laguna State Polytechnic University.*
