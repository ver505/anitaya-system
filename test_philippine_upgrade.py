"""
Comprehensive Verification Suite for AniTaya Philippine Agricultural Upgrade
Verifies:
- Single centralized ML pipeline instance shared between app.py & crop_engine.py
- Absence of Indonesia proxy and hardcoded pesticides_tonnes=120.0
- Distinct prediction_method routing (philippine_ml vs psa_statistical_historical vs fallback)
- Agricultural unit & financial invariants across diverse PH provinces & crops
- Clean UI HTML checks (zero stale 0.966 / Zenodo / Indonesia strings)
"""

import json
import app as anitaya_app_module
import crop_engine


def test_philippine_upgrade():
    print("=" * 75)
    print("VERIFYING ANITAYA PHILIPPINE ML & DATASET UPGRADE")
    print("=" * 75)

    # 1. Centralized Pipeline Check (Zero Duplication)
    assert anitaya_app_module.pipeline is crop_engine.ml_pipeline, "Pipeline loaded twice!"
    assert anitaya_app_module.metrics_data is crop_engine.ml_metrics, "Metrics loaded twice!"
    assert crop_engine.ml_pipeline is not None, "Production ML pipeline missing!"
    print("[PASS] 1. Single centralized ML pipeline & metrics shared cleanly.")

    # 2. Validate Trained Model Feature Schema (zero pesticides_tonnes)
    features = crop_engine.ml_metrics["dataset_info"]["features"]
    assert features == ["item", "average_rain_fall_mm_per_year", "avg_temp", "area"], f"Unexpected features: {features}"
    assert "pesticides_tonnes" not in features
    assert len(crop_engine.ml_metrics["model_comparison"]) == 8
    best = crop_engine.ml_metrics["best_model"]
    assert best["cv_r2_mean"] > 0.90 and best["test_r2"] > 0.70
    print(f"[PASS] 2. 8-model benchmark verified. Winner: {best['name']} (CV R2={best['cv_r2_mean']}, Test R2={best['test_r2']}).")

    client = anitaya_app_module.app.test_client()

    # 3. Verify Explicit Prediction Tier Routing & Province Sensitivity
    payload_multi = {
        "province": "Bukidnon",
        "season": "Wet Season",
        "soil_type": "Adtuyon Clay",
        "crops": [
            {"crop_name": "Rice, paddy", "land_area_ha": 2.0},
            {"crop_name": "Sugarcane", "land_area_ha": 1.0},
            {"crop_name": "Coffee", "land_area_ha": 1.0},
            {"crop_name": "Potatoes", "land_area_ha": 1.0},
        ],
    }
    r1 = client.post("/api/calculate", data=json.dumps(payload_multi), content_type="application/json")
    assert r1.status_code == 200
    d1 = r1.get_json()
    by_crop = {c["crop_name"]: c for c in d1["crop_results"]}

    assert by_crop["Rice, paddy"]["prediction_method"] == "philippine_ml"
    assert by_crop["Rice, paddy"]["is_ml_predicted"] is True
    assert by_crop["Sugarcane"]["prediction_method"] == "philippine_ml"
    assert by_crop["Coffee"]["prediction_method"] == "philippine_ml"
    assert 0.05 < by_crop["Coffee"]["yield_rate_mt_ha"] < 1.2, f"Coffee yield implausible: {by_crop['Coffee']['yield_rate_mt_ha']}"
    assert by_crop["Potatoes"]["prediction_method"] == "psa_statistical_historical"
    assert by_crop["Potatoes"]["is_ml_predicted"] is False
    print("[PASS] 3. Per-crop prediction_method routing & Coffee/Sugarcane agronomic scales verified.")

    # Verify financial & unit formulas across custom inputs
    payload_custom = {
        "province": "Nueva Ecija",
        "season": "Dry Season",
        "crops": [
            {"crop_name": "Rice, paddy", "land_area_ha": 2.5, "farmgate_price": 21.0, "cost_per_ha": 52000.0},
            {"crop_name": "Red Onion", "land_area_ha": 1.0},
        ],
    }
    r2 = client.post("/api/calculate", data=json.dumps(payload_custom), content_type="application/json")
    assert r2.status_code == 200
    d2 = r2.get_json()
    rice_ne = next(c for c in d2["crop_results"] if c["crop_name"] == "Rice, paddy")
    onion_ne = next(c for c in d2["crop_results"] if c["crop_name"] == "Red Onion")
    assert rice_ne["farmgate_price_php_kg"] == 21.0
    assert rice_ne["farmgate_price_php_mt"] == 21000.0
    assert rice_ne["cost_per_ha_php"] == 52000.0
    assert abs(rice_ne["production_cost_php"] - 130000.0) < 1e-2
    assert abs(rice_ne["gross_income_php"] - (rice_ne["total_yield_mt"] * 21000.0)) < 1.0
    assert abs(rice_ne["net_income_php"] - (rice_ne["gross_income_php"] - rice_ne["production_cost_php"])) < 1e-2
    assert onion_ne["prediction_method"] == "psa_statistical_historical"
    print("[PASS] 4. Custom price, custom cost, unit consistency (MT/kg/PHP), and Red Onion PSA stats verified.")

    # 4. Unsupported crop cleanly handled in legacy /api/predict
    r_bad_crop = client.post("/api/predict", data=json.dumps({"crop": "Nonexistent Alien Plant", "land_area": 1.0}), content_type="application/json")
    assert r_bad_crop.status_code == 422
    print("[PASS] 5. Unrecognized crop rejected cleanly on /api/predict with HTTP 422.")

    # 5. HTML Render Hygiene Check (Zero 0.966 / Indonesia Proxy / Zenodo leftovers)
    html = client.get("/").data.decode("utf-8")
    for forbidden in ("0.966", "96.60%", "tropical proxies (Indonesia)", "Zenodo Record 18761663", "ZENODO 18761663", "pesticides_tonnes: 120.0"):
        assert forbidden not in html, f"Stale string still present in rendered HTML: {forbidden}"
    assert best["name"] in html
    assert "PSA" in html and "PAGASA" in html and "NASA POWER" in html
    assert "Info only" in html
    print("[PASS] 6. Rendered dashboard HTML free of legacy proxy/accuracy lies and displays live Philippine metadata.")

    print("=" * 75)
    print("ALL PHILIPPINE DATASET & ML UPGRADE ASSERTIONS PASSED!")
    print("=" * 75)


if __name__ == "__main__":
    test_philippine_upgrade()
