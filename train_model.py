import os
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "df_yield_merged.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
METRICS_PATH = os.path.join(MODELS_DIR, "model_metrics.json")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "anitaya_best_pipeline.joblib")

def log(msg):
    print(msg, flush=True)

def train_and_evaluate():
    log(f"Loading dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    log(f"Dataset shape: {df.shape}")

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    feature_cols = ["item", "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp", "area"]
    target_col = "value_hg_ha"

    X = df[feature_cols]
    y = df[target_col]

    log("\nSplitting dataset (80% train, 20% test, random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, shuffle=True
    )
    log(f"Training samples: {len(X_train):,}, Testing samples: {len(X_test):,}")

    categorical_features = ["item", "area"]
    numeric_features = ["average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]

    # Preprocessing Pipeline using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )

    candidate_models = {
        "Linear Regression (Baseline)": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=15, random_state=42),
        "Random Forest (Primary Model)": RandomForestRegressor(
            n_estimators=100, max_depth=20, random_state=42, n_jobs=1
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=100, max_depth=5, random_state=42
        )
    }

    results = {}
    best_model_name = None
    best_r2 = -float("inf")
    best_pipeline = None

    log("\n" + "="*85)
    log(f"{'Model':<32} | {'MAE (hg/ha)':<12} | {'RMSE (hg/ha)':<13} | {'R² Score':<10} | {'3-Fold CV R²':<12}")
    log("="*85)

    # Use a representative subset of train for CV to ensure rapid evaluation, or full
    cv = KFold(n_splits=3, shuffle=True, random_state=42)

    for name, model in candidate_models.items():
        t0 = time.time()
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model)])

        # Train model
        log(f"Training {name}...")
        pipeline.fit(X_train, y_train)

        # Predict on Test set
        y_pred = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Cross Validation (3-fold, sequential)
        cv_scores = cross_val_score(pipeline, X_train.iloc[:5000], y_train.iloc[:5000], cv=cv, scoring="r2", n_jobs=1)
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))

        train_time = time.time() - t0

        results[name] = {
            "MAE": round(float(mae), 2),
            "MSE": round(float(mse), 2),
            "RMSE": round(float(rmse), 2),
            "R2": round(float(r2), 4),
            "CV_R2_Mean": round(cv_mean, 4),
            "CV_R2_Std": round(cv_std, 4),
            "Training_Time_Sec": round(train_time, 2)
        }

        log(f"-> {name:<30} | {mae:<12.1f} | {rmse:<13.1f} | {r2:<10.4f} | {cv_mean:.4f} (±{cv_std:.3f}) [{train_time:.1f}s]")

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_pipeline = pipeline

    log("="*85)
    log(f"Selected Final Model: {best_model_name} with Test R² = {best_r2:.4f}")

    # Feature Importance analysis
    feature_importances = []
    if hasattr(best_pipeline.named_steps["regressor"], "feature_importances_"):
        rf_regressor = best_pipeline.named_steps["regressor"]
        cat_encoder = best_pipeline.named_steps["preprocessor"].named_transformers_["cat"]
        cat_feature_names = list(cat_encoder.get_feature_names_out(categorical_features))
        all_feature_names = numeric_features + cat_feature_names
        importances = rf_regressor.feature_importances_
        
        aggregated = {
            "Crop Type (item)": 0.0,
            "Country / Region (area)": 0.0,
            "Rainfall (average_rain_fall_mm_per_year)": 0.0,
            "Pesticides / Agrochemical Usage (pesticides_tonnes)": 0.0,
            "Average Temperature (avg_temp)": 0.0,
        }
        for fname, imp in zip(all_feature_names, importances):
            if fname.startswith("item_"):
                aggregated["Crop Type (item)"] += float(imp)
            elif fname.startswith("area_"):
                aggregated["Country / Region (area)"] += float(imp)
            elif fname == "average_rain_fall_mm_per_year":
                aggregated["Rainfall (average_rain_fall_mm_per_year)"] += float(imp)
            elif fname == "pesticides_tonnes":
                aggregated["Pesticides / Agrochemical Usage (pesticides_tonnes)"] += float(imp)
            elif fname == "avg_temp":
                aggregated["Average Temperature (avg_temp)"] += float(imp)

        for k, v in aggregated.items():
            feature_importances.append({"feature": k, "importance": round(v * 100, 2)})
        feature_importances.sort(key=lambda x: x["importance"], reverse=True)

    # Philippine Agricultural Benchmarks (PHP / kg and PHP / ha)
    ph_benchmarks = {
        "Rice, paddy": {
            "filipino_name": "Palay (Palay / Bigas)",
            "farmgate_price_per_kg": 22.0,
            "cost_per_ha": 45000.0,
            "typical_yield_mt_ha": 4.2,
            "cavan_weight_kg": 50,
            "growing_days": 115,
            "recommended_season": "Wet & Dry Season",
            "production_cost_breakdown": {
                "Seeds": 12.0,
                "Fertilizers & Agrochemicals": 32.0,
                "Labor & Land Preparation": 38.0,
                "Fuel, Irrigation & Threshing": 18.0
            }
        },
        "Maize": {
            "filipino_name": "Mais (Corn / Yellow & White)",
            "farmgate_price_per_kg": 20.5,
            "cost_per_ha": 38000.0,
            "typical_yield_mt_ha": 4.5,
            "cavan_weight_kg": 50,
            "growing_days": 105,
            "recommended_season": "Dry Season / Late Wet",
            "production_cost_breakdown": {
                "Seeds": 15.0,
                "Fertilizers & Agrochemicals": 35.0,
                "Labor & Land Preparation": 35.0,
                "Machinery & Fuel": 15.0
            }
        },
        "Cassava": {
            "filipino_name": "Kamoteng Kahoy (Cassava)",
            "farmgate_price_per_kg": 14.0,
            "cost_per_ha": 32000.0,
            "typical_yield_mt_ha": 12.5,
            "cavan_weight_kg": 50,
            "growing_days": 270,
            "recommended_season": "Early Wet Season",
            "production_cost_breakdown": {
                "Planting Stems": 10.0,
                "Fertilizers": 25.0,
                "Labor & Weeding": 50.0,
                "Harvesting & Transport": 15.0
            }
        },
        "Potatoes": {
            "filipino_name": "Patatas (Irish Potato)",
            "farmgate_price_per_kg": 45.0,
            "cost_per_ha": 95000.0,
            "typical_yield_mt_ha": 18.0,
            "cavan_weight_kg": 50,
            "growing_days": 90,
            "recommended_season": "Cool Dry Season",
            "production_cost_breakdown": {
                "Seed Tubers": 40.0,
                "Fertilizers & Crop Care": 30.0,
                "Labor": 20.0,
                "Storage & Transport": 10.0
            }
        },
        "Sweet potatoes": {
            "filipino_name": "Kamote (Sweet Potato)",
            "farmgate_price_per_kg": 28.0,
            "cost_per_ha": 28000.0,
            "typical_yield_mt_ha": 8.0,
            "cavan_weight_kg": 50,
            "growing_days": 110,
            "recommended_season": "All Seasons",
            "production_cost_breakdown": {
                "Cuttings / Vines": 10.0,
                "Fertilizer": 25.0,
                "Labor": 45.0,
                "Harvesting": 20.0
            }
        },
        "Plantains and others": {
            "filipino_name": "Saging (Banana / Saba / Plantains)",
            "farmgate_price_per_kg": 25.0,
            "cost_per_ha": 40000.0,
            "typical_yield_mt_ha": 15.0,
            "cavan_weight_kg": 50,
            "growing_days": 365,
            "recommended_season": "Year-round",
            "production_cost_breakdown": {
                "Suckers / Seedlings": 15.0,
                "Fertilizer & Mulch": 30.0,
                "Maintenance & De-suckering": 35.0,
                "Harvesting": 20.0
            }
        },
        "Soybeans": {
            "filipino_name": "Utaw (Soybeans)",
            "farmgate_price_per_kg": 35.0,
            "cost_per_ha": 30000.0,
            "typical_yield_mt_ha": 2.0,
            "cavan_weight_kg": 50,
            "growing_days": 100,
            "recommended_season": "Dry Season",
            "production_cost_breakdown": {
                "Seeds": 15.0,
                "Inoculants & Fertilizer": 25.0,
                "Labor": 40.0,
                "Post-harvest": 20.0
            }
        },
        "Sorghum": {
            "filipino_name": "Batad (Sorghum)",
            "farmgate_price_per_kg": 18.0,
            "cost_per_ha": 25000.0,
            "typical_yield_mt_ha": 3.0,
            "cavan_weight_kg": 50,
            "growing_days": 100,
            "recommended_season": "Dry / Drought-tolerant",
            "production_cost_breakdown": {
                "Seeds": 12.0,
                "Fertilizers": 28.0,
                "Labor": 40.0,
                "Threshing": 20.0
            }
        },
        "Yams": {
            "filipino_name": "Ube (Purple Yam)",
            "farmgate_price_per_kg": 65.0,
            "cost_per_ha": 55000.0,
            "typical_yield_mt_ha": 9.0,
            "cavan_weight_kg": 50,
            "growing_days": 240,
            "recommended_season": "Wet Season",
            "production_cost_breakdown": {
                "Setts / Seed Pieces": 30.0,
                "Trellising & Poles": 20.0,
                "Fertilizer": 20.0,
                "Labor & Digging": 30.0
            }
        },
        "Wheat": {
            "filipino_name": "Trigo (Wheat)",
            "farmgate_price_per_kg": 24.0,
            "cost_per_ha": 35000.0,
            "typical_yield_mt_ha": 3.2,
            "cavan_weight_kg": 50,
            "growing_days": 110,
            "recommended_season": "Cool Season",
            "production_cost_breakdown": {
                "Seeds": 15.0,
                "Fertilizers": 35.0,
                "Labor": 30.0,
                "Harvesting": 20.0
            }
        }
    }

    os.makedirs(MODELS_DIR, exist_ok=True)
    log(f"\nSaving best pipeline to: {BEST_MODEL_PATH}")
    joblib.dump(best_pipeline, BEST_MODEL_PATH)

    output_metadata = {
        "dataset_info": {
            "source": "Zenodo Record 18761663 (DOI: 10.5281/zenodo.18761663)",
            "total_records": len(df),
            "training_samples": len(X_train),
            "testing_samples": len(X_test),
            "features": feature_cols,
            "target": target_col,
            "crops_count": len(df["item"].unique()),
            "crops": sorted(list(df["item"].unique())),
            "countries_count": len(df["area"].unique()),
            "years": [int(df["year"].min()), int(df["year"].max())]
        },
        "model_comparison": results,
        "best_model": {
            "name": best_model_name,
            "test_r2": results[best_model_name]["R2"],
            "test_mae": results[best_model_name]["MAE"],
            "test_rmse": results[best_model_name]["RMSE"],
            "cv_r2_mean": results[best_model_name]["CV_R2_Mean"],
            "cv_r2_std": results[best_model_name]["CV_R2_Std"]
        },
        "feature_importances": feature_importances,
        "ph_benchmarks": ph_benchmarks
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(output_metadata, f, indent=4)
    log(f"Metrics and metadata saved to: {METRICS_PATH}")
    log("\nTraining and evaluation successfully completed!")

if __name__ == "__main__":
    train_and_evaluate()
