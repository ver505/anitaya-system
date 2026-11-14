import os
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
import joblib

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
MODEL_PATH = os.path.join(BASE_DIR, "models", "anitaya_best_pipeline.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "models", "model_metrics.json")
PROVINCES_PATH = os.path.join(BASE_DIR, "data", "philippine_provinces_crop_suitability.json")

# Initialize SQLite Prediction History Database
try:
    history_db.init_db()
    print("Taya: SQLite Prediction History database initialized.")
except Exception as e:
    print(f"Taya Warning: Could not initialize database: {e}")

# Load model pipeline and metrics
pipeline = None
metrics_data = {}
provinces_data = []
provinces_by_name = {}
provinces_by_region = {}

try:
    pipeline = joblib.load(MODEL_PATH)
    print("Taya: Machine Learning pipeline loaded successfully.")
except Exception as e:
    print(f"Taya Warning: Could not load pipeline: {e}")

try:
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)
    print("Taya: Metrics and Philippine benchmarks loaded successfully.")
except Exception as e:
    print(f"Taya Warning: Could not load metrics: {e}")

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

    rec_str = (p.get("primary_crops", "") + "; " + p.get("secondary_crops", "")).lower()
    annotated = []
    for crop in crop_engine.crop_catalog:
        c_name = crop["Crop_Name"].lower()
        is_rec = (c_name in rec_str) or (crop.get("Crop_ID", "").lower() in rec_str) or (crop_engine.normalize_crop_name(c_name).lower() in rec_str)
        suitable_provs = str(crop.get("Suitable_Provinces", "")).lower()
        is_prov_suitable = p["province"].lower() in suitable_provs

        annotated.append({
            **crop,
            "is_recommended": is_rec,
            "is_suitable": is_rec or is_prov_suitable,
            "suitability_grade": p.get("da_suitability_grade", "Suitable") if is_rec else "Secondary Option"
        })

    return jsonify({
        "status": "success",
        "province": p["province"],
        "region": p["region"],
        "count": len(annotated),
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
    stats["total_crops"] = len(crop_engine.crop_catalog)
    stats["total_provinces"] = len(provinces_data)
    stats["model_name"] = metrics_data.get("best_model", {}).get("name", "Random Forest Regressor")
    stats["model_r2"] = metrics_data.get("best_model", {}).get("test_r2", 0.966)
    return jsonify({"status": "success", "stats": stats})

@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Legacy Single-Crop Prediction Route (Preserved 100% Backwards-Compatible).
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"status": "error", "message": "No input payload provided."}), 400

        # Extract and validate inputs
        raw_crop = data.get("crop", "Rice").strip()
        CROP_NAME_MAP = {
            "rice": "Rice, paddy",
            "rice, paddy": "Rice, paddy",
            "palay": "Rice, paddy",
            "corn": "Maize",
            "maize": "Maize",
            "mais": "Maize",
            "potato": "Potatoes",
            "potatoes": "Potatoes",
            "patatas": "Potatoes",
            "cassava": "Cassava",
            "kamoteng kahoy": "Cassava"
        }
        ml_crop = CROP_NAME_MAP.get(raw_crop.lower(), raw_crop)
        crop = ml_crop
        land_area = float(data.get("land_area", 1.0))
        season = data.get("season", "Wet Season").strip()
        region = data.get("region", "Laguna / CALABARZON (Region IV-A)")
        province_name = data.get("province", "").strip()

        # Check if province matches official DA database
        matched_province = None
        if province_name:
            matched_province = provinces_by_name.get(province_name.lower())
            if not matched_province:
                matches = [p for p in provinces_data if province_name.lower() in p["province"].lower()]
                if matches:
                    matched_province = matches[0]

        # Auto-resolve climate parameters from selected province or region
        if matched_province:
            default_rainfall = matched_province["annual_rainfall_mm"]
            default_temp = matched_province["avg_temp_c"]
            region = matched_province["region"]
        else:
            reg_info = REGIONAL_PRESETS.get(region, REGIONAL_PRESETS["Laguna / CALABARZON (Region IV-A)"])
            default_rainfall = reg_info["rainfall"]
            default_temp = reg_info["temp"]

        rainfall = float(data.get("rainfall") if data.get("rainfall") is not None else default_rainfall)
        avg_temp = float(data.get("avg_temp") if data.get("avg_temp") is not None else default_temp)
        pesticides_tonnes = float(data.get("pesticides_tonnes") if data.get("pesticides_tonnes") is not None else 120.0)

        custom_price = data.get("farmgate_price")
        custom_cost = data.get("cost_per_ha")

        # Range validations
        errors = []
        if land_area <= 0 or land_area > 500:
            errors.append("Land area must be greater than 0 and less than 500 hectares.")
        if rainfall < 100 or rainfall > 6000:
            errors.append("Annual rainfall must be between 100 mm and 6,000 mm.")
        if avg_temp < 10 or avg_temp > 48:
            errors.append("Average temperature must be between 10°C and 48°C.")
        if pesticides_tonnes < 0 or pesticides_tonnes > 5000:
            errors.append("Pesticides/agrochemical input must be non-negative.")

        benchmarks = metrics_data.get("ph_benchmarks", {}).get(crop)
        if not benchmarks:
            errors.append(f"Selected crop '{crop}' is not supported in legacy benchmarks.")

        if errors:
            return jsonify({"status": "error", "message": "Validation failed", "errors": errors}), 422

        # Seasonal adjustment
        effective_rainfall = rainfall
        if season == "Wet Season":
            effective_rainfall = max(rainfall, 1800.0)
        else:
            effective_rainfall = min(rainfall, 1600.0)

        # Prepare input for ML model
        input_df = pd.DataFrame([{
            "item": crop,
            "average_rain_fall_mm_per_year": effective_rainfall,
            "pesticides_tonnes": pesticides_tonnes,
            "avg_temp": avg_temp,
            "area": "Indonesia"
        }])

        predicted_hg_ha = float(pipeline.predict(input_df)[0])
        predicted_hg_ha = max(predicted_hg_ha, 5000.0)

        yield_rate_mt_ha = predicted_hg_ha / 10000.0
        yield_rate_kg_ha = predicted_hg_ha / 10.0

        total_yield_kg = yield_rate_kg_ha * land_area
        total_yield_mt = total_yield_kg / 1000.0
        cavan_weight = benchmarks.get("cavan_weight_kg", 50)
        total_sacks = total_yield_kg / cavan_weight

        # Financial Calculations (in Philippine Peso ₱ - PHP)
        farmgate_price = float(custom_price) if custom_price and float(custom_price) > 0 else benchmarks["farmgate_price_per_kg"]
        cost_per_ha = float(custom_cost) if custom_cost and float(custom_cost) > 0 else benchmarks["cost_per_ha"]

        gross_revenue = total_yield_kg * farmgate_price
        total_production_cost = cost_per_ha * land_area
        net_profit = gross_revenue - total_production_cost
        profit_margin = (net_profit / gross_revenue * 100.0) if gross_revenue > 0 else 0.0
        roi = (net_profit / total_production_cost * 100.0) if total_production_cost > 0 else 0.0

        cost_shares = {
            "Seeds & Planting Materials": 15.0,
            "Fertilizers & Agrochemicals": 30.0,
            "Labor & Land Preparation": 35.0,
            "Fuel, Irrigation & Machinery": 20.0
        }
        cost_breakdown = {}
        for category, share in cost_shares.items():
            cost_breakdown[category] = round((share / 100.0) * total_production_cost, 2)

        scenarios = {
            "favorable": {
                "label": "Favorable Season (+15%)",
                "condition": "Optimal rainfall distribution, zero pest pressure, high solar hours",
                "yield_mt": round(total_yield_mt * 1.15, 2),
                "yield_kg": round(total_yield_kg * 1.15, 1),
                "sacks": round(total_sacks * 1.15, 1),
                "gross_revenue": round(gross_revenue * 1.15, 2),
                "net_profit": round((gross_revenue * 1.15) - total_production_cost, 2)
            },
            "expected": {
                "label": "Expected Baseline (Normal)",
                "condition": "Standard seasonal conditions and regular farm inputs",
                "yield_mt": round(total_yield_mt, 2),
                "yield_kg": round(total_yield_kg, 1),
                "sacks": round(total_sacks, 1),
                "gross_revenue": round(gross_revenue, 2),
                "net_profit": round(net_profit, 2)
            },
            "adverse": {
                "label": "Adverse Shock (-20%)",
                "condition": "Monsoon waterlogging, moderate pest outbreak, or dry spell",
                "yield_mt": round(total_yield_mt * 0.80, 2),
                "yield_kg": round(total_yield_kg * 0.80, 1),
                "sacks": round(total_sacks * 0.80, 1),
                "gross_revenue": round(gross_revenue * 0.80, 2),
                "net_profit": round((gross_revenue * 0.80) - total_production_cost, 2)
            }
        }

        insights = []
        if crop == "Rice, paddy":
            if season == "Wet Season":
                insights.append("During the Wet Season, palay benefits from abundant rainfall. Ensure proper dike drainage to avoid submergence during heading stages.")
            else:
                insights.append("During the Dry Season, higher solar radiation enhances grain density. Ensure scheduled canal irrigation to maximize yields.")
        elif crop == "Maize":
            if season == "Wet Season":
                insights.append("During the Wet Season, protect corn furrows from water stagnation to minimize root rot and leaf fungal blight.")
            else:
                insights.append("Dry Season is optimal for Maize production! Grain drying is faster and farmgate market demand is typically peak.")
        elif crop == "Potatoes":
            insights.append("Irish Potatoes thrive best in cool highland climates like Benguet. Maintain balanced potassium fertilizer and monitor for late blight.")
        elif crop == "Cassava":
            insights.append("Cassava is hardy and drought-resilient. Harvest between 8 to 10 months to achieve peak root starch yield.")
        else:
            insights.append(f"Optimal conditions detected for {benchmarks['filipino_name']}. Maintain balanced soil nutrients and weed control.")

        response_payload = {
            "status": "success",
            "inputs": {
                "crop": crop,
                "crop_filipino": benchmarks["filipino_name"],
                "land_area_ha": land_area,
                "season": season,
                "region": region,
                "rainfall_mm": rainfall,
                "pesticides_tonnes": pesticides_tonnes,
                "avg_temp_c": avg_temp,
                "farmgate_price_per_kg": farmgate_price,
                "cost_per_ha": cost_per_ha
            },
            "yield_predictions": {
                "yield_rate_mt_ha": round(yield_rate_mt_ha, 2),
                "yield_rate_kg_ha": round(yield_rate_kg_ha, 1),
                "total_yield_mt": round(total_yield_mt, 2),
                "expected_production_mt": round(total_yield_mt, 2),
                "total_yield_kg": round(total_yield_kg, 1),
                "total_sacks_cavans": round(total_sacks, 1),
                "cavan_weight_kg": cavan_weight,
                "growing_period_days": benchmarks["growing_days"]
            },
            "financial_forecast": {
                "farmgate_price_used": farmgate_price,
                "gross_revenue_php": round(gross_revenue, 2),
                "production_cost_php": round(total_production_cost, 2),
                "net_profit_php": round(net_profit, 2),
                "profit_margin_percent": round(profit_margin, 2),
                "roi_percent": round(roi, 2),
                "cost_breakdown": cost_breakdown
            },
            "scenarios": scenarios,
            "insights": insights,
            "model_metadata": {
                "model_name": metrics_data.get("best_model", {}).get("name", "Random Forest Regressor"),
                "model_r2": metrics_data.get("best_model", {}).get("test_r2", 0.966),
                "model_mae": metrics_data.get("best_model", {}).get("test_mae", 6908.5)
            }
        }

        if matched_province:
            is_rec = crop.lower() in matched_province["primary_crops"].lower()
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
                "agronomic_advice": matched_province["agronomic_advice"]
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
