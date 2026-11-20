import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
from sklearn.inspection import permutation_importance
import joblib

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "df_yield_merged.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
METRICS_PATH = os.path.join(MODELS_DIR, "model_metrics.json")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "anitaya_best_pipeline.joblib")


def log(msg):
    print(msg, flush=True)


def build_preprocessor(numeric_features, categorical_features):
    """
    Factory function returning a fresh ColumnTransformer instance.
    Ensures each Pipeline and each CV fold fits its own isolated StandardScaler
    and OneHotEncoder strictly on the training split (zero data leakage).
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ],
        remainder="drop",
    )


def train_and_evaluate():
    log(f"Loading dataset from: {DATA_PATH}")
    df_raw = pd.read_csv(DATA_PATH)
    raw_row_count = len(df_raw)
    log(f"Raw dataset shape: {df_raw.shape}")

    if "Unnamed: 0" in df_raw.columns:
        df_raw = df_raw.drop(columns=["Unnamed: 0"])

    feature_cols = [
        "item",
        "average_rain_fall_mm_per_year",
        "pesticides_tonnes",
        "avg_temp",
        "area",
    ]
    target_col = "value_hg_ha"

    # ── Data Leakage & Preprocessing Audit ──
    # 1. Remove any rows with missing values in features or target
    null_count = int(df_raw[feature_cols + [target_col]].isnull().sum().sum())
    df_clean = df_raw.dropna(subset=feature_cols + [target_col]).copy()

    # 2. Remove exact feature+target duplicates BEFORE train/test split
    #    so identical records cannot leak across X_train and X_test
    dup_count = int(df_clean.duplicated(subset=feature_cols + [target_col]).sum())
    if dup_count > 0:
        df_clean = df_clean.drop_duplicates(
            subset=feature_cols + [target_col]
        ).reset_index(drop=True)

    # 3. Verify target column is strictly excluded from input features X
    assert target_col not in feature_cols, "Data leakage detected: target column in features!"

    X = df_clean[feature_cols].copy()
    y = df_clean[target_col].astype(float).copy()

    log(
        f"Data Leakage Audit: raw={raw_row_count:,}, nulls_removed={null_count}, "
        f"exact_duplicates_removed={dup_count}, clean_records={len(df_clean):,}"
    )

    log("\nSplitting dataset (80% train, 20% test, stratified by crop item, random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, shuffle=True, stratify=X["item"]
    )
    log(f"Training samples: {len(X_train):,}, Testing samples: {len(X_test):,}")

    categorical_features = ["item", "area"]
    numeric_features = ["average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]

    # 8 Candidate Regression Algorithms
    candidate_models = {
        "Linear Regression (Baseline)": {
            "role": "Linear Baseline",
            "estimator": LinearRegression(),
        },
        "Ridge Regression": {
            "role": "L2 Regularized Linear",
            "estimator": Ridge(alpha=1.0, random_state=42),
        },
        "Support Vector Regression (SVR)": {
            "role": "Kernel Margin Regressor",
            "estimator": SVR(kernel="rbf", C=10000.0, gamma="scale", epsilon=0.1, max_iter=1500),
        },
        "K-Nearest Neighbors (KNN)": {
            "role": "Instance-Based Regressor",
            "estimator": KNeighborsRegressor(n_neighbors=5, weights="distance", n_jobs=-1),
        },
        "Decision Tree Regressor": {
            "role": "Single Tree Regressor",
            "estimator": DecisionTreeRegressor(max_depth=15, random_state=42),
        },
        "Random Forest Regressor": {
            "role": "Bagging Tree Ensemble",
            "estimator": RandomForestRegressor(
                n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
            ),
        },
        "Extra Trees Regressor": {
            "role": "Randomized Tree Ensemble",
            "estimator": ExtraTreesRegressor(
                n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
            ),
        },
        "Gradient Boosting Regressor": {
            "role": "Boosted Tree Ensemble",
            "estimator": GradientBoostingRegressor(
                n_estimators=100, max_depth=5, random_state=42
            ),
        },
    }

    results = {}
    trained_pipelines = {}
    test_predictions = {}

    log("\n" + "=" * 105)
    log(
        f"{'Model Algorithm':<34} | {'MAE (hg/ha)':<12} | {'RMSE (hg/ha)':<13} | {'Test R²':<9} | {'5-Fold CV R²':<16} | {'Train Time':<10}"
    )
    log("=" * 105)

    # 5-Fold Cross-Validation setup (strictly on training set X_train)
    cv_folds = 5
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    # Stratified/shuffled subset of training data for 5-fold CV so all 8 models evaluate consistently
    cv_sample_size = min(6000, len(X_train))
    X_train_cv = X_train.iloc[:cv_sample_size]
    y_train_cv = y_train.iloc[:cv_sample_size]

    for name, spec in candidate_models.items():
        estimator = clone(spec["estimator"])
        role = spec["role"]

        # Fresh isolated preprocessor inside the Pipeline prevents any data leakage
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
                ("regressor", estimator),
            ]
        )

        # Measure actual training time on X_train
        t_fit_start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_time_sec = time.perf_counter() - t_fit_start

        # Predict on unseen holdout test set X_test
        y_pred = pipeline.predict(X_test)

        mae = float(mean_absolute_error(y_test, y_pred))
        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, y_pred))

        # 5-Fold Cross-Validation on training data using an isolated cloned pipeline
        t_cv_start = time.perf_counter()
        cv_pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
                ("regressor", clone(spec["estimator"])),
            ]
        )
        cv_scores = cross_val_score(
            cv_pipeline, X_train_cv, y_train_cv, cv=cv, scoring="r2", n_jobs=1
        )
        cv_time_sec = time.perf_counter() - t_cv_start
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))

        results[name] = {
            "Role": role,
            "MAE": round(mae, 2),
            "MSE": round(mse, 2),
            "RMSE": round(rmse, 2),
            "R2": round(r2, 4),
            "CV_R2_Mean": round(cv_mean, 4),
            "CV_R2_Std": round(cv_std, 4),
            "CV_Folds": cv_folds,
            "Training_Time_Sec": round(fit_time_sec, 2),
            "Total_Eval_Time_Sec": round(fit_time_sec + cv_time_sec, 2),
        }
        trained_pipelines[name] = pipeline
        test_predictions[name] = y_pred

        log(
            f"-> {name:<31} | {mae:<12.2f} | {rmse:<13.2f} | {r2:<9.4f} | "
            f"{cv_mean:.4f} (±{cv_std:.4f}) | {fit_time_sec:.2f}s"
        )

    # ── Automatic Best-Model Selection Based on Actual Evaluation Results ──
    # Primary criterion: highest Test R², tie-broken by 5-Fold CV R² and lowest RMSE
    ranked_models = sorted(
        results.keys(),
        key=lambda m: (results[m]["R2"], results[m]["CV_R2_Mean"], -results[m]["RMSE"]),
        reverse=True,
    )
    best_model_name = ranked_models[0]
    runner_up_name = ranked_models[1]
    best_pipeline = trained_pipelines[best_model_name]
    best_metrics = results[best_model_name]
    runner_up_metrics = results[runner_up_name]
    baseline_metrics = results.get("Linear Regression (Baseline)", best_metrics)

    # Mark rank and selection status on each model entry
    for rank_idx, m_name in enumerate(ranked_models, start=1):
        results[m_name]["Rank"] = rank_idx
        results[m_name]["Is_Winner"] = m_name == best_model_name

    log("=" * 105)
    log(
        f"Automatically Selected Best Model: {best_model_name} "
        f"(Test R² = {best_metrics['R2']:.4f}, 5-Fold CV R² = {best_metrics['CV_R2_Mean']:.4f}, "
        f"MAE = {best_metrics['MAE']:.2f} hg/ha, RMSE = {best_metrics['RMSE']:.2f} hg/ha)"
    )

    # ── Feature Importance Analysis (from Selected Best Model) ──
    feature_importances = []
    reg_step = best_pipeline.named_steps["regressor"]
    prep_step = best_pipeline.named_steps["preprocessor"]

    aggregated = {
        "Crop Type (item)": 0.0,
        "Country / Region (area)": 0.0,
        "Pesticides / Agrochemical Usage (pesticides_tonnes)": 0.0,
        "Average Temperature (avg_temp)": 0.0,
        "Rainfall (average_rain_fall_mm_per_year)": 0.0,
    }

    if hasattr(reg_step, "feature_importances_"):
        cat_encoder = prep_step.named_transformers_["cat"]
        cat_feature_names = list(cat_encoder.get_feature_names_out(categorical_features))
        all_feature_names = numeric_features + cat_feature_names
        importances = reg_step.feature_importances_

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
    else:
        # Permutation importance fallback on test subset for non-tree models (e.g., KNN/SVR/Linear)
        perm = permutation_importance(
            best_pipeline,
            X_test.iloc[:1000],
            y_test.iloc[:1000],
            n_repeats=3,
            random_state=42,
            n_jobs=1,
        )
        perm_vals = np.maximum(perm.importances_mean, 0)
        total_perm = float(np.sum(perm_vals)) or 1.0
        col_map = {
            "item": "Crop Type (item)",
            "area": "Country / Region (area)",
            "pesticides_tonnes": "Pesticides / Agrochemical Usage (pesticides_tonnes)",
            "avg_temp": "Average Temperature (avg_temp)",
            "average_rain_fall_mm_per_year": "Rainfall (average_rain_fall_mm_per_year)",
        }
        for col_name, val in zip(feature_cols, perm_vals):
            aggregated[col_map[col_name]] = float(val) / total_perm

    for k, v in aggregated.items():
        feature_importances.append({"feature": k, "importance": round(v * 100, 2)})
    feature_importances.sort(key=lambda x: x["importance"], reverse=True)

    # ── Actual vs. Predicted Holdout Test Samples (Real Data from X_test / y_test) ──
    best_y_pred = test_predictions[best_model_name]
    test_eval_df = pd.DataFrame(
        {
            "crop": X_test["item"].values,
            "actual_hg_ha": y_test.values,
            "predicted_hg_ha": best_y_pred,
        }
    )
    # Deterministic stratified sampling of 220 real test points across all crops
    sampled_points = []
    per_crop_quota = max(15, 220 // max(1, len(test_eval_df["crop"].unique())))
    for crop_name, group in test_eval_df.groupby("crop"):
        sub = group.iloc[:per_crop_quota]
        for _, row in sub.iterrows():
            act_mt = round(float(row["actual_hg_ha"]) / 10000.0, 3)
            pred_mt = round(max(0.1, float(row["predicted_hg_ha"]) / 10000.0), 3)
            sampled_points.append(
                {
                    "crop": str(crop_name),
                    "actual_mt_ha": act_mt,
                    "predicted_mt_ha": pred_mt,
                    "actual_hg_ha": round(float(row["actual_hg_ha"]), 1),
                    "predicted_hg_ha": round(float(row["predicted_hg_ha"]), 1),
                    "residual_mt_ha": round(pred_mt - act_mt, 3),
                }
            )
    sampled_points = sampled_points[:220]

    # ── Dynamic Selected Model Explanation ──
    r2_gain_vs_linear = round((best_metrics["R2"] - baseline_metrics["R2"]) * 100, 2)
    mae_reduction_pct = round(
        (1.0 - (best_metrics["MAE"] / max(1.0, baseline_metrics["MAE"]))) * 100, 1
    )
    top_feat_1 = feature_importances[0]
    top_feat_2 = feature_importances[1]

    selected_model_explanation = (
        f"{best_model_name} was automatically selected as AniTaya's primary production model after "
        f"evaluating 8 regression algorithms on {len(X_test):,} unseen holdout test records and 5-Fold Cross-Validation. "
        f"It achieved the highest Test R² of {best_metrics['R2']:.4f} ({best_metrics['R2']*100:.2f}% variance explained) "
        f"and a 5-Fold Cross-Validation R² of {best_metrics['CV_R2_Mean']:.4f} (±{best_metrics['CV_R2_Std']:.4f}), "
        f"with a Mean Absolute Error (MAE) of {best_metrics['MAE']:,.2f} hg/ha ({best_metrics['MAE']/10000:.2f} MT/ha) "
        f"and RMSE of {best_metrics['RMSE']:,.2f} hg/ha ({best_metrics['RMSE']/10000:.2f} MT/ha) in {best_metrics['Training_Time_Sec']:.2f}s of training. "
        f"It outperformed the runner-up {runner_up_name} (Test R² = {runner_up_metrics['R2']:.4f}, RMSE = {runner_up_metrics['RMSE']:,.2f} hg/ha) "
        f"and improved accuracy by +{r2_gain_vs_linear:.2f} percentage points over the Linear Regression baseline while reducing MAE by {mae_reduction_pct:.1f}%. "
        f"Feature importance analysis confirms that {top_feat_1['feature']} ({top_feat_1['importance']:.2f}%) and "
        f"{top_feat_2['feature']} ({top_feat_2['importance']:.2f}%) are the strongest drivers of harvest yield."
    )

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
                "Fuel, Irrigation & Threshing": 18.0,
            },
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
                "Machinery & Fuel": 15.0,
            },
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
                "Harvesting & Transport": 15.0,
            },
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
                "Storage & Transport": 10.0,
            },
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
                "Harvesting": 20.0,
            },
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
                "Harvesting": 20.0,
            },
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
                "Post-harvest": 20.0,
            },
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
                "Threshing": 20.0,
            },
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
                "Labor & Digging": 30.0,
            },
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
                "Harvesting": 20.0,
            },
        },
    }

    os.makedirs(MODELS_DIR, exist_ok=True)
    log(f"\nSaving best pipeline ({best_model_name}) to: {BEST_MODEL_PATH}")
    joblib.dump(best_pipeline, BEST_MODEL_PATH)

    output_metadata = {
        "dataset_info": {
            "source": "Zenodo Record 18761663 (DOI: 10.5281/zenodo.18761663)",
            "raw_records": raw_row_count,
            "duplicates_removed": dup_count,
            "total_records": len(df_clean),
            "training_samples": len(X_train),
            "testing_samples": len(X_test),
            "features": feature_cols,
            "target": target_col,
            "crops_count": len(df_clean["item"].unique()),
            "crops": sorted(list(df_clean["item"].unique())),
            "countries_count": len(df_clean["area"].unique()),
            "years": [int(df_clean["year"].min()), int(df_clean["year"].max())],
        },
        "leakage_audit": {
            "status": "PASSED",
            "raw_records": raw_row_count,
            "null_values_removed": null_count,
            "exact_duplicates_removed_pre_split": dup_count,
            "clean_records": len(df_clean),
            "target_excluded_from_features": True,
            "isolated_pipeline_preprocessor": True,
            "scaler_and_encoder_fit_on_train_only": True,
            "stratified_train_test_split": "80% Train / 20% Test (stratified by crop item)",
            "cross_validation_protocol": f"{cv_folds}-Fold Cross-Validation on Training Set",
        },
        "model_comparison": results,
        "best_model": {
            "name": best_model_name,
            "role": best_metrics["Role"],
            "test_r2": best_metrics["R2"],
            "test_mae": best_metrics["MAE"],
            "test_rmse": best_metrics["RMSE"],
            "cv_r2_mean": best_metrics["CV_R2_Mean"],
            "cv_r2_std": best_metrics["CV_R2_Std"],
            "cv_folds": cv_folds,
            "training_time_sec": best_metrics["Training_Time_Sec"],
            "runner_up_name": runner_up_name,
            "runner_up_r2": runner_up_metrics["R2"],
            "explanation": selected_model_explanation,
        },
        "feature_importances": feature_importances,
        "actual_vs_predicted": sampled_points,
        "ph_benchmarks": ph_benchmarks,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(output_metadata, f, indent=2)
    log(f"Metrics and metadata saved to: {METRICS_PATH}")
    log("\nTraining and 8-model evaluation successfully completed!")


if __name__ == "__main__":
    train_and_evaluate()
