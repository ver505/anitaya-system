import os
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

import history_db
import crop_engine

app = Flask(__name__)

# Disable all caching in Flask and Jinja2 so edits show immediately
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.jinja_env.auto_reload = True

@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROVINCES_PATH = os.path.join(BASE_DIR, "data", "philippine_provinces_crop_suitability.json")

# Initialize SQLite Prediction History Database
try:
    history_db.init_db()
    print("Taya: SQLite Prediction History database initialized.")
except Exception as e:
    print(f"Taya Warning: Could not initialize database: {e}")

# Centralized single-load ML pipeline & metrics shared directly from crop_engine (zero duplicate load)
pipeline = crop_engine.ml_pipeline
metrics_data = crop_engine.ml_metrics if isinstance(crop_engine.ml_metrics, dict) else {}
provinces_data = []
provinces_by_name = {}
provinces_by_region = {}

# Load Official Philippine Provincial Crop Suitability & Recommendation Dataset (DA-AMIA & PSA)
try:
    if os.path.exists(PROVINCES_PATH):
        with open(PROVINCES_PATH, "r", encoding="utf-8") as f:
            provinces_data = json.load(f)
        for p in provinces_data:
            provinces_by_name[p["province"].strip().lower()] = p
            reg = p["region"]
            if reg not in provinces_by_region:
                provinces_by_region[reg] = []
            provinces_by_region[reg].append(p)
        print(f"Taya: Loaded {len(provinces_data)} Philippine provinces with official DA-AMIA & PSA crop recommendations.")
except Exception as e:
    print(f"Taya Warning: Could not load provincial suitability data: {e}")

# Philippine Regional Climate Presets
REGIONAL_PRESETS = {
    "Laguna / CALABARZON (Region IV-A)": {"rainfall": 2400.0, "temp": 28.2, "desc": "Tropical rainforest climate, wet monsoon peaks from July to November"},
    "Central Luzon (Region III - Rice Granary)": {"rainfall": 1950.0, "temp": 28.5, "desc": "Distinct wet and dry seasons, prime zone for Rice and Corn"},
    "Ilocos Region (Region I)": {"rainfall": 1800.0, "temp": 28.0, "desc": "Pronounced dry season (Nov-Apr), wet season (May-Oct)"},
    "Cagayan Valley (Region II)": {"rainfall": 2100.0, "temp": 27.5, "desc": "Fertile river basin, major producer of corn, rice, and cassava"},
    "Bicol Region (Region V)": {"rainfall": 3100.0, "temp": 27.8, "desc": "High annual rainfall; requires resilient cassava and robust drainage"},
    "Western Visayas (Region VI)": {"rainfall": 2350.0, "temp": 28.1, "desc": "Rich alluvial plains, strong rice and corn production"},
    "Davao Region (Region XI)": {"rainfall": 1900.0, "temp": 28.4, "desc": "Even rainfall distribution year-round, typhoon-free tropical belt"},
    "Northern Mindanao (Region X)": {"rainfall": 2200.0, "temp": 27.2, "desc": "Highland and plateau agriculture, high corn and cassava yields"},
    "Cordillera / Benguet Highlands": {"rainfall": 2600.0, "temp": 19.5, "desc": "Cool mountain climate, premier producing region for Irish Potatoes"}
}

@app.route("/")
def index():
    crops = metrics_data.get("ph_benchmarks", {})
    model_comparison = metrics_data.get("model_comparison", {})
    best_model = metrics_data.get("best_model", {})
    feature_importances = metrics_data.get("feature_importances", [])
    dataset_info = metrics_data.get("dataset_info", {})
    actual_vs_predicted = metrics_data.get("actual_vs_predicted", [])
    leakage_audit = metrics_data.get("leakage_audit", {})
    system_stats = history_db.get_stats()
    system_stats["total_crops"] = len(crop_engine.crop_catalog)
    system_stats["total_provinces"] = len(provinces_data)

    return render_template(
        "index.html",
        crops=crops,
        regional_presets=REGIONAL_PRESETS,
        provinces_data=provinces_data,
        provinces_by_region=provinces_by_region,
        model_comparison=model_comparison,
        best_model=best_model,
        feature_importances=feature_importances,
        dataset_info=dataset_info,
        actual_vs_predicted=actual_vs_predicted,
        leakage_audit=leakage_audit,
        metrics_data=metrics_data,
        crops_catalog=crop_engine.crop_catalog,
        system_stats=system_stats
    )

@app.route("/api/benchmarks", methods=["GET"])
def get_benchmarks():
    return jsonify({
        "crops": metrics_data.get("ph_benchmarks", {}),
        "regions": REGIONAL_PRESETS
    })

@app.route("/api/provinces", methods=["GET"])
def get_provinces():
    return jsonify({
        "status": "success",
        "count": len(provinces_data),
        "authority": "Department of Agriculture (DA-AMIA NCCAG v2.0) & Philippine Statistics Authority (PSA)",
        "data": provinces_data
    })

@app.route("/api/province/<province_name>", methods=["GET"])
def get_province(province_name):
    query = province_name.strip().lower()
    p = provinces_by_name.get(query)
    if not p:
        matches = [item for item in provinces_data if query in item["province"].lower()]
        if matches:
            p = matches[0]
    if p:
        return jsonify({"status": "success", "data": p})
    return jsonify({"status": "error", "message": f"Province '{province_name}' not found in official DA-AMIA catalog."}), 404

@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    return jsonify(metrics_data)

@app.route("/api/crops", methods=["GET"])
def get_crops():
    """Returns the comprehensive Philippine Crop Database (35+ official crops)."""
    return jsonify({
        "status": "success",
        "count": len(crop_engine.crop_catalog),
        "source": "Philippine Statistics Authority (PSA) & Bureau of Plant Industry (DA-BPI)",
        "data": crop_engine.crop_catalog
    })

@app.route("/api/crops/suitability/<province_name>", methods=["GET"])
def get_crops_by_suitability(province_name):
    """Returns crops annotated with suitability status for the specified Philippine province."""
    query = province_name.strip().lower()
    p = provinces_by_name.get(query)
    if not p:
        for k, v in provinces_by_name.items():
            if query in k or k in query:
                p = v
                break

    if not p:
        return jsonify({"status": "error", "message": f"Province '{province_name}' not found."}), 404

    prov_psa_summary = crop_engine.get_province_psa_crop_summary(p["province"])
    annotated = []
    for crop in crop_engine.crop_catalog:
        c_name = crop["Crop_Name"]
        is_suitable, is_da_rec, suitability_notes = crop_engine.assess_crop_province_suitability(
            c_name, p, crop
        )
        psa_info = prov_psa_summary.get(c_name, {})
        pm_info = crop_engine.per_crop_metrics_map.get(c_name, {})

        annotated.append({
            **crop,
            "is_recommended": is_suitable,
            "is_da_recommended": is_da_rec,
            "is_suitable": is_suitable,
            "suitability_notes": suitability_notes,
            "suitability_grade": p.get("da_suitability_grade", "Suitable") if is_suitable else "Secondary Option",
            "psa_historical_yield_mt_ha": psa_info.get("recent_yield_mt_ha"),
            "psa_latest_production_mt": psa_info.get("latest_production_mt"),
            "psa_latest_year": psa_info.get("latest_year"),
            "prediction_method": pm_info.get("prediction_tier", crop.get("Prediction_Method", "psa_statistical_historical")),
        })

    return jsonify({
        "status": "success",
        "province": p["province"],
        "region": p["region"],
        "count": len(annotated),
        "psa_observed_crops_count": len(prov_psa_summary),
        "data": annotated
    })

@app.route("/api/calculate", methods=["POST"])
def calculate_multi_crop():
    """
    Primary Multi-Crop Agricultural Prediction & Financial Engine.
    Accepts multiple crops with individual land areas and farming conditions.
    Calculates estimated Production, Farmgate Prices, Gross Income, Costs, and Net Income.
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"status": "error", "message": "No input payload provided."}), 400

        result, code = crop_engine.process_multi_crop_calculation(data, provinces_by_name)
        return jsonify(result), code
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/history", methods=["GET"])
def list_history():
    """Retrieves recent saved predictions from SQLite database."""
    try:
        records = history_db.get_history(limit=50)
        return jsonify({"status": "success", "count": len(records), "history": records})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/history/<int:record_id>", methods=["GET"])
def get_history_detail(record_id):
    """Retrieves detailed record of a saved prediction."""
    try:
        rec = history_db.get_prediction_detail(record_id)
        if rec:
            return jsonify({"status": "success", "data": rec})
        return jsonify({"status": "error", "message": f"Prediction record #{record_id} not found."}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/history/save", methods=["POST"])
def save_prediction_route():
    """Saves a completed multi-crop calculation to SQLite."""
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"status": "error", "message": "No data to save."}), 400
        rec_id = history_db.save_prediction(data)
        return jsonify({"status": "success", "id": rec_id, "message": "Prediction saved successfully."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/history/<int:record_id>", methods=["DELETE"])
def delete_history_route(record_id):
    """Deletes a specific prediction from SQLite."""
    try:
        deleted = history_db.delete_prediction(record_id)
        if deleted:
            return jsonify({"status": "success", "message": f"Prediction #{record_id} deleted."})
        return jsonify({"status": "error", "message": "Record not found or already deleted."}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/history/clear", methods=["POST"])
def clear_all_history_route():
    """Clears all prediction history from SQLite."""
    try:
        history_db.clear_all_history()
        return jsonify({"status": "success", "message": "All prediction history cleared."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/stats", methods=["GET"])
def get_stats_route():
    """Returns dynamic system counters."""
    stats = history_db.get_stats()
    best_info = metrics_data.get("best_model", {})
    ds_info = metrics_data.get("dataset_info", {})
    stats["total_crops"] = len(crop_engine.crop_catalog)
    stats["total_provinces"] = len(provinces_data)
    stats["model_name"] = best_info.get("name", "Extra Trees Regressor")
    stats["model_r2"] = best_info.get("test_r2")
    stats["cv_r2_mean"] = best_info.get("cv_r2_mean")
    stats["model_mae_hg_ha"] = best_info.get("test_mae")
    stats["ml_crops_count"] = ds_info.get("ml_supported_crops_count", len(crop_engine.ml_supported_crops))
    stats["statistical_crops_count"] = ds_info.get("statistical_fallback_crops_count", 0)
    return jsonify({"status": "success", "stats": stats})

@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Legacy Single-Crop Prediction Route (Backwards-Compatible Contract).
    Uses real Philippine provinces and reliable 4-feature climate+crop pipeline
    (zero Indonesia proxy, zero hardcoded pesticides_tonnes=120.0).
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"status": "error", "message": "No input payload provided."}), 400

        raw_crop = str(data.get("crop", "Rice, paddy")).strip()
        crop = crop_engine.normalize_crop_name(raw_crop)
        land_area = float(data.get("land_area", 1.0))
        season = str(data.get("season", "Wet Season")).strip()
        region = str(data.get("region", "Laguna / CALABARZON (Region IV-A)")).strip()
        province_name = str(data.get("province", "")).strip()

        matched_province = None
        if province_name:
            matched_province = provinces_by_name.get(province_name.lower())
            if not matched_province:
                matches = [p for p in provinces_data if province_name.lower() in p["province"].lower()]
                if matches:
                    matched_province = matches[0]
        elif region in crop_engine.REGION_REP_PROVINCE:
            rep_name = crop_engine.REGION_REP_PROVINCE[region]
            matched_province = provinces_by_name.get(rep_name.lower())

        if matched_province:
            resolved_province = matched_province["province"]
            default_rainfall = matched_province["annual_rainfall_mm"]
            default_temp = matched_province["avg_temp_c"]
            region = matched_province["region"]
        else:
            reg_info = REGIONAL_PRESETS.get(region, REGIONAL_PRESETS["Laguna / CALABARZON (Region IV-A)"])
            resolved_province = crop_engine.REGION_REP_PROVINCE.get(region, "Laguna")
            default_rainfall = reg_info["rainfall"]
            default_temp = reg_info["temp"]

        rainfall = float(data.get("rainfall") if data.get("rainfall") is not None else default_rainfall)
        avg_temp = float(data.get("avg_temp") if data.get("avg_temp") is not None else default_temp)
        custom_price = data.get("farmgate_price")
        custom_cost = data.get("cost_per_ha")

        errors = []
        if land_area <= 0 or land_area > 500:
            errors.append("Land area must be greater than 0 and less than 500 hectares.")
        if rainfall < 100 or rainfall > 6000:
            errors.append("Annual rainfall must be between 100 mm and 6,000 mm.")
        if avg_temp < 10 or avg_temp > 48:
            errors.append("Average temperature must be between 10°C and 48°C.")

        crop_meta = crop_engine.get_crop_info(crop)
        benchmarks = metrics_data.get("ph_benchmarks", {}).get(crop)
        if not crop_meta and not benchmarks:
            errors.append(f"Selected crop '{crop}' is not recognized in the Philippine crop database.")

        if errors:
            return jsonify({"status": "error", "message": "Validation failed", "errors": errors}), 422

        crop_calc_input = {
            "crop_name": crop,
            "land_area_ha": land_area,
            "farmgate_price": custom_price,
            "cost_per_ha": custom_cost,
        }
        # Override matched_province climate with explicit request rainfall/temp when provided
        sim_prov_info = dict(matched_province) if matched_province else {"province": resolved_province, "region": region}
        sim_prov_info["annual_rainfall_mm"] = rainfall
        sim_prov_info["avg_temp_c"] = avg_temp

        single_res = crop_engine.calculate_single_crop(
            crop_calc_input,
            sim_prov_info,
            season=season,
            soil_type=str(data.get("soil_type", sim_prov_info.get("dominant_soil", "Loam"))),
        )

        yield_rate_mt_ha = single_res["yield_rate_mt_ha"]
        yield_rate_kg_ha = single_res["yield_rate_kg_ha"]
        total_yield_mt = single_res["total_yield_mt"]
        total_yield_kg = single_res["total_yield_kg"]
        cavan_weight = benchmarks.get("cavan_weight_kg", 50) if benchmarks else 50
        total_sacks = round(total_yield_kg / cavan_weight, 1)

        farmgate_price = single_res["farmgate_price_php_kg"]
        cost_per_ha = single_res["cost_per_ha_php"]
        gross_revenue = single_res["gross_income_php"]
        total_production_cost = single_res["production_cost_php"]
        net_profit = single_res["net_income_php"]
        profit_margin = single_res["profit_margin_percent"]
        roi = single_res["roi_percent"]

        cost_breakdown = {
            "Seeds & Planting Materials": single_res["itemized_costs"]["seeds_cost_php"],
            "Fertilizers & Crop Care": single_res["itemized_costs"]["fertilizer_cost_php"] + single_res["itemized_costs"]["pesticide_cost_php"],
            "Labor & Land Preparation": single_res["itemized_costs"]["labor_cost_php"],
            "Fuel, Irrigation & Other": single_res["itemized_costs"]["other_costs_php"],
        }

        scenarios = {
            k: {
                "label": sc["label"],
                "condition": sc["condition"],
                "yield_mt": sc["yield_mt"],
                "yield_kg": sc["yield_kg"],
                "sacks": round(sc["yield_kg"] / cavan_weight, 1),
                "gross_revenue": sc["gross_income_php"],
                "net_profit": sc["net_income_php"],
            }
            for k, sc in single_res["scenarios"].items()
        }

        insights = [single_res["suitability_notes"]]
        if crop == "Rice, paddy":
            insights.append(
                "Wet Season Palay benefits from monsoon rain; maintain proper dike drainage during reproductive heading."
                if "wet" in season.lower()
                else "Dry Season solar radiation increases grain filling when supplemental canal irrigation is scheduled."
            )
        elif crop == "Maize":
            insights.append(
                "Protect corn furrows against waterlogging during Wet Season rains."
                if "wet" in season.lower()
                else "Dry Season supports higher grain quality and post-harvest sun drying for Maize."
            )
        elif crop == "Potatoes":
            insights.append("Highland potatoes thrive in cool mountain climates such as Benguet; scout regularly for late blight.")
        elif crop == "Cassava":
            insights.append("Cassava is drought-tolerant but requires well-drained soils to prevent root rot.")

        best_meta = metrics_data.get("best_model", {})
        fil_label = benchmarks.get("filipino_name", crop) if benchmarks else crop
        g_days = benchmarks.get("growing_days", 120) if benchmarks else 120

        response_payload = {
            "status": "success",
            "inputs": {
                "crop": crop,
                "crop_filipino": fil_label,
                "land_area_ha": land_area,
                "season": season,
                "province": resolved_province,
                "region": region,
                "rainfall_mm": rainfall,
                "avg_temp_c": avg_temp,
                "farmgate_price_per_kg": farmgate_price,
                "cost_per_ha": cost_per_ha,
            },
            "yield_predictions": {
                "yield_rate_mt_ha": yield_rate_mt_ha,
                "yield_rate_kg_ha": yield_rate_kg_ha,
                "total_yield_mt": total_yield_mt,
                "expected_production_mt": total_yield_mt,
                "total_yield_kg": total_yield_kg,
                "total_sacks_cavans": total_sacks,
                "cavan_weight_kg": cavan_weight,
                "growing_period_days": g_days,
                "prediction_method": single_res["prediction_method"],
                "yield_source": single_res["yield_source"],
                "is_ml_predicted": single_res["is_ml_predicted"],
            },
            "financial_forecast": {
                "farmgate_price_used": farmgate_price,
                "price_unit": single_res["price_unit"],
                "price_source": single_res["price_source"],
                "cost_source": single_res["cost_source"],
                "gross_revenue_php": gross_revenue,
                "production_cost_php": total_production_cost,
                "net_profit_php": net_profit,
                "profit_margin_percent": profit_margin,
                "roi_percent": roi,
                "cost_breakdown": cost_breakdown,
            },
            "scenarios": scenarios,
            "insights": insights,
            "model_metadata": {
                "model_name": best_meta.get("name", "Extra Trees Regressor"),
                "model_r2": best_meta.get("test_r2"),
                "cv_r2_mean": best_meta.get("cv_r2_mean"),
                "model_mae": best_meta.get("test_mae"),
                "prediction_method": single_res["prediction_method"],
            },
        }

        if matched_province:
            is_rec, _da_rec, _notes = crop_engine.assess_crop_province_suitability(crop, matched_province)
            response_payload["province_suitability"] = {
                "province": matched_province["province"],
                "region": matched_province["region"],
                "is_recommended": is_rec,
                "da_grade": matched_province["da_suitability_grade"],
                "primary_crops": matched_province["primary_crops"],
                "secondary_crops": matched_province["secondary_crops"],
                "soil_type": matched_province["dominant_soil"],
                "climate_type": matched_province["climate_type"],
                "annual_rainfall_mm": matched_province["annual_rainfall_mm"],
                "avg_temp_c": matched_province["avg_temp_c"],
                "elevation": matched_province["elevation_category"],
                "national_distinction": matched_province["national_distinction"],
                "agronomic_advice": matched_province["agronomic_advice"],
            }

        return jsonify(response_payload)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Taya Web Server at http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
