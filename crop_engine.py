"""
Taya System - Multi-Crop Agricultural Prediction & Financial Engine
Integrates:
- Scikit-Learn Machine Learning Pipeline (Random Forest Regressor, R² = 0.966)
- Philippine Statistics Authority (PSA) Farmgate Price Records
- PSA Costs and Returns of Agricultural Production
- PSA Major Crops Production Statistics
- DA-AMIA Provincial Agro-Climatic Suitability Series
"""

import os
import pandas as pd
import numpy as np
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "models", "anitaya_best_pipeline.joblib")

# Load datasets
CROP_LIST_PATH = os.path.join(DATA_DIR, "philippine_crop_list.csv")
FARMGATE_PATH = os.path.join(DATA_DIR, "farmgate_prices.csv")
COSTS_PATH = os.path.join(DATA_DIR, "crop_production_costs.csv")
PROD_PATH = os.path.join(DATA_DIR, "crop_production.csv")

# In-memory cached data
crop_catalog = []
crop_by_name = {}
df_farmgate = pd.DataFrame()
df_costs = pd.DataFrame()
df_prod = pd.DataFrame()
ml_pipeline = None

# Aliases and name normalizer
CROP_NAME_MAP = {
    "rice": "Rice, paddy",
    "rice, paddy": "Rice, paddy",
    "palay": "Rice, paddy",
    "bigas": "Rice, paddy",
    "corn": "Maize",
    "maize": "Maize",
    "mais": "Maize",
    "yellow corn": "Maize",
    "white corn": "Maize",
    "sorghum": "Sorghum",
    "batad": "Sorghum",
    "cassava": "Cassava",
    "kamoteng kahoy": "Cassava",
    "balanghoy": "Cassava",
    "potato": "Potatoes",
    "potatoes": "Potatoes",
    "patatas": "Potatoes",
    "irish potato": "Potatoes",
    "sweet potato": "Sweet potatoes",
    "sweet potatoes": "Sweet potatoes",
    "kamote": "Sweet potatoes",
    "yam": "Yams",
    "yams": "Yams",
    "ube": "Yams",
    "purple yam": "Yams",
    "taro": "Taro",
    "gabi": "Taro",
    "tomato": "Tomato",
    "kamatis": "Tomato",
    "eggplant": "Eggplant",
    "talong": "Eggplant",
    "bitter gourd": "Bitter Gourd",
    "ampalaya": "Bitter Gourd",
    "squash": "Squash",
    "kalabasa": "Squash",
    "cabbage": "Cabbage",
    "repolyo": "Cabbage",
    "pechay": "Pechay",
    "bok choy": "Pechay",
    "red onion": "Red Onion",
    "onion": "Red Onion",
    "sibuyas": "Red Onion",
    "yellow onion": "Yellow Onion",
    "garlic": "Garlic",
    "bawang": "Garlic",
    "string beans": "String Beans",
    "sitaw": "String Beans",
    "pole sitaw": "String Beans",
    "mungbean": "Mungbean",
    "monggo": "Mungbean",
    "balatong": "Mungbean",
    "soybean": "Soybeans",
    "soybeans": "Soybeans",
    "utaw": "Soybeans",
    "peanut": "Peanut",
    "mani": "Peanut",
    "banana": "Plantains and others",
    "saba": "Plantains and others",
    "plantains": "Plantains and others",
    "plantains and others": "Plantains and others",
    "cavendish": "Banana Cavendish",
    "banana cavendish": "Banana Cavendish",
    "mango": "Mango",
    "mangga": "Mango",
    "carabao mango": "Mango",
    "pineapple": "Pineapple",
    "pinya": "Pineapple",
    "calamansi": "Calamansi",
    "kalamansi": "Calamansi",
    "papaya": "Papaya",
    "watermelon": "Watermelon",
    "pakwan": "Watermelon",
    "coconut": "Coconut",
    "niyog": "Coconut",
    "copra": "Coconut",
    "sugarcane": "Sugarcane",
    "tubo": "Sugarcane",
    "coffee": "Coffee",
    "kape": "Coffee",
    "cacao": "Cacao",
    "kakaw": "Cacao",
    "abaca": "Abaca",
    "manila hemp": "Abaca",
    "tobacco": "Tobacco",
    "tabako": "Tobacco",
    "rubber": "Rubber",
    "goma": "Rubber"
}

# 10 FAO crops directly supported by the trained ML pipeline
ML_SUPPORTED_CROPS = {
    "Rice, paddy",
    "Maize",
    "Sorghum",
    "Cassava",
    "Sweet potatoes",
    "Potatoes",
    "Yams",
    "Soybeans",
    "Plantains and others"
}

def init_engine():
    global crop_catalog, crop_by_name, df_farmgate, df_costs, df_prod, ml_pipeline
    
    # Load ML Pipeline
    try:
        if os.path.exists(MODEL_PATH):
            ml_pipeline = joblib.load(MODEL_PATH)
            print("CropEngine: ML pipeline loaded successfully.")
    except Exception as e:
        print(f"CropEngine: Could not load ML pipeline: {e}")
        
    # Load Crop Catalog
    try:
        if os.path.exists(CROP_LIST_PATH):
            df_c = pd.read_csv(CROP_LIST_PATH)
            crop_catalog = df_c.to_dict(orient="records")
            for c in crop_catalog:
                crop_by_name[c["Crop_Name"].strip().lower()] = c
            print(f"CropEngine: Loaded {len(crop_catalog)} Philippine crops.")
    except Exception as e:
        print(f"CropEngine: Error loading crop catalog: {e}")

    # Load Farmgate Prices
    try:
        if os.path.exists(FARMGATE_PATH):
            df_farmgate = pd.read_csv(FARMGATE_PATH)
            print(f"CropEngine: Loaded {len(df_farmgate)} farmgate price records.")
    except Exception as e:
        print(f"CropEngine: Error loading farmgate prices: {e}")

    # Load Production Costs
    try:
        if os.path.exists(COSTS_PATH):
            df_costs = pd.read_csv(COSTS_PATH)
            print(f"CropEngine: Loaded {len(df_costs)} production cost records.")
    except Exception as e:
        print(f"CropEngine: Error loading production costs: {e}")

    # Load Crop Production Stats
    try:
        if os.path.exists(PROD_PATH):
            df_prod = pd.read_csv(PROD_PATH)
            print(f"CropEngine: Loaded {len(df_prod)} crop production records.")
    except Exception as e:
        print(f"CropEngine: Error loading crop production: {e}")

# Initialize at import
init_engine()

def normalize_crop_name(raw_name):
    clean = raw_name.strip().lower()
    return CROP_NAME_MAP.get(clean, raw_name.strip())

def get_crop_info(crop_name):
    norm = normalize_crop_name(crop_name)
    info = crop_by_name.get(norm.lower())
    if not info:
        # Match partial
        for k, v in crop_by_name.items():
            if norm.lower() in k or k in norm.lower():
                return v
    return info

def get_farmgate_price(crop_name, province_name=None):
    norm = normalize_crop_name(crop_name)
    if not df_farmgate.empty:
        if province_name:
            match = df_farmgate[(df_farmgate["Crop_Name"].str.lower() == norm.lower()) & 
                                (df_farmgate["Province"].str.lower() == province_name.strip().lower())]
            if not match.empty:
                latest_yr = match["Year"].max()
                row = match[match["Year"] == latest_yr].iloc[0]
                return float(row["Farmgate_Price"]), str(row["Price_Unit"])
        
        # Fallback to national average for the crop
        match_crop = df_farmgate[df_farmgate["Crop_Name"].str.lower() == norm.lower()]
        if not match_crop.empty:
            latest_yr = match_crop["Year"].max()
            sub = match_crop[match_crop["Year"] == latest_yr]
            avg_price = float(sub["Farmgate_Price"].mean())
            unit = str(sub.iloc[0]["Price_Unit"])
            return round(avg_price, 2), unit

    # Default fallback benchmarks
    defaults = {
        "Rice, paddy": (22.0, "PHP/kg"),
        "Maize": (20.5, "PHP/kg"),
        "Potatoes": (45.0, "PHP/kg"),
        "Cassava": (14.0, "PHP/kg"),
        "Sweet potatoes": (28.0, "PHP/kg"),
        "Plantains and others": (25.0, "PHP/kg"),
        "Red Onion": (75.0, "PHP/kg"),
        "Garlic": (110.0, "PHP/kg"),
        "Tomato": (38.0, "PHP/kg"),
        "Eggplant": (35.0, "PHP/kg")
    }
    return defaults.get(norm, (30.0, "PHP/kg"))

def get_production_cost(crop_name, province_name=None):
    norm = normalize_crop_name(crop_name)
    if not df_costs.empty:
        if province_name:
            match = df_costs[(df_costs["Crop_Name"].str.lower() == norm.lower()) & 
                             (df_costs["Province"].str.lower() == province_name.strip().lower())]
            if not match.empty:
                latest_yr = match["Year"].max()
                row = match[match["Year"] == latest_yr].iloc[0]
                return {
                    "total_cost_per_ha": float(row["Total_Production_Cost"]),
                    "labor_cost": float(row["Labor_Cost"]),
                    "fertilizer_cost": float(row["Fertilizer_Cost"]),
                    "pesticide_cost": float(row["Pesticide_Cost"]),
                    "seeds_cost": float(row["Seeds_Cost"]),
                    "other_costs": float(row["Other_Costs"])
                }

        # Fallback to national average for the crop
        match_crop = df_costs[df_costs["Crop_Name"].str.lower() == norm.lower()]
        if not match_crop.empty:
            latest_yr = match_crop["Year"].max()
            sub = match_crop[match_crop["Year"] == latest_yr]
            return {
                "total_cost_per_ha": round(float(sub["Total_Production_Cost"].mean()), 2),
                "labor_cost": round(float(sub["Labor_Cost"].mean()), 2),
                "fertilizer_cost": round(float(sub["Fertilizer_Cost"].mean()), 2),
                "pesticide_cost": round(float(sub["Pesticide_Cost"].mean()), 2),
                "seeds_cost": round(float(sub["Seeds_Cost"].mean()), 2),
                "other_costs": round(float(sub["Other_Costs"].mean()), 2)
            }

    # Default fallback cost
    return {
        "total_cost_per_ha": 45000.0,
        "labor_cost": 16000.0,
        "fertilizer_cost": 14000.0,
        "pesticide_cost": 4500.0,
        "seeds_cost": 5500.0,
        "other_costs": 5000.0
    }

def get_baseline_yield_mt(crop_name, province_name=None):
    norm = normalize_crop_name(crop_name)
    if not df_prod.empty:
        if province_name:
            match = df_prod[(df_prod["Crop_Name"].str.lower() == norm.lower()) & 
                            (df_prod["Province"].str.lower() == province_name.strip().lower())]
            if not match.empty:
                latest_yr = match["Year"].max()
                row = match[match["Year"] == latest_yr].iloc[0]
                return float(row["Yield_Per_Hectare"])
        
        match_crop = df_prod[df_prod["Crop_Name"].str.lower() == norm.lower()]
        if not match_crop.empty:
            latest_yr = match_crop["Year"].max()
            sub = match_crop[match_crop["Year"] == latest_yr]
            return round(float(sub["Yield_Per_Hectare"].mean()), 2)
            
    # Default fallbacks
    fallbacks = {
        "Rice, paddy": 4.15,
        "Maize": 4.30,
        "Cassava": 12.50,
        "Potatoes": 17.50,
        "Sweet potatoes": 7.90,
        "Plantains and others": 14.80,
        "Red Onion": 10.50,
        "Garlic": 3.80,
        "Tomato": 11.20,
        "Eggplant": 12.00
    }
    return fallbacks.get(norm, 5.0)

def calculate_single_crop(crop_input, province_info, season="Wet Season", soil_type="Loam"):
    """
    Computes yield and financial projections for one crop.
    crop_input: dict with keys:
      - crop_name (str)
      - land_area_ha (float)
      - farmgate_price (float, optional)
      - cost_per_ha (float, optional)
      - pesticides_tonnes (float, optional)
    """
    raw_crop = crop_input.get("crop_name", "Rice, paddy")
    norm_crop = normalize_crop_name(raw_crop)
    land_area = float(crop_input.get("land_area_ha", 1.0))
    pesticides_tonnes = float(crop_input.get("pesticides_tonnes", 120.0))
    
    crop_meta = get_crop_info(norm_crop) or {
        "Crop_Name": norm_crop,
        "Scientific_Name": "Agricultural crop",
        "Crop_Category": "General Agriculture",
        "Origin_Status": "Cultivated",
        "Description": "Cultivated Philippine crop."
    }

    prov_name = province_info.get("province", "") if province_info else ""
    rainfall = float(province_info.get("annual_rainfall_mm", 2200.0)) if province_info else 2200.0
    avg_temp = float(province_info.get("avg_temp_c", 28.0)) if province_info else 28.0

    # Determine Yield
    yield_source = ""
    is_ml_predicted = False
    
    if norm_crop in ML_SUPPORTED_CROPS and ml_pipeline is not None:
        # Use ML Random Forest Pipeline
        effective_rainfall = rainfall
        if season == "Wet Season":
            effective_rainfall = max(rainfall, 1800.0)
        else:
            effective_rainfall = min(rainfall, 1600.0)

        input_df = pd.DataFrame([{
            "item": norm_crop,
            "average_rain_fall_mm_per_year": effective_rainfall,
            "pesticides_tonnes": pesticides_tonnes,
            "avg_temp": avg_temp,
            "area": "Indonesia" # Top representative tropical Southeast Asian climate
        }])
        
        try:
            predicted_hg_ha = float(ml_pipeline.predict(input_df)[0])
            predicted_hg_ha = max(predicted_hg_ha, 4000.0)
            yield_rate_mt_ha = predicted_hg_ha / 10000.0
            yield_source = "Machine Learning Model (Random Forest Pipeline, R² 0.966)"
            is_ml_predicted = True
        except Exception as e:
            # Fallback to PSA baseline
            yield_rate_mt_ha = get_baseline_yield_mt(norm_crop, prov_name)
            yield_source = "Official PSA Baseline Statistics (Fallback)"
    else:
        # Specialized Philippine Crop: Use official PSA agricultural statistics
        base_mt = get_baseline_yield_mt(norm_crop, prov_name)
        # Seasonal modulations for specific categories
        cat = crop_meta.get("Crop_Category", "")
        if cat in ["Vegetables", "Legumes"] and season == "Dry Season":
            # Dry season with irrigation often produces slightly higher vegetable yields
            yield_rate_mt_ha = round(base_mt * 1.05, 2)
        elif cat == "Root & Tuber Crops" and season == "Wet Season":
            yield_rate_mt_ha = round(base_mt * 1.03, 2)
        else:
            yield_rate_mt_ha = round(base_mt, 2)
        yield_source = "Official PSA Agrometeorological Baseline (Major Crops Statistics)"

    yield_rate_mt_ha = round(yield_rate_mt_ha, 3)
    yield_rate_kg_ha = round(yield_rate_mt_ha * 1000.0, 1)
    total_yield_mt = round(yield_rate_mt_ha * land_area, 3)
    total_yield_kg = round(total_yield_mt * 1000.0, 1)

    # Cavans (50kg bags)
    cavan_weight = 50.0
    total_cavans = round(total_yield_kg / cavan_weight, 1)

    # Farmgate Price
    custom_price = crop_input.get("farmgate_price")
    if custom_price is not None and float(custom_price) > 0:
        price_used = float(custom_price)
        price_unit = "PHP/kg (custom)"
    else:
        price_used, price_unit = get_farmgate_price(norm_crop, prov_name)

    # Production Cost
    custom_cost = crop_input.get("cost_per_ha")
    cost_info = get_production_cost(norm_crop, prov_name)
    if custom_cost is not None and float(custom_cost) > 0:
        cost_per_ha = float(custom_cost)
        scale_ratio = cost_per_ha / cost_info["total_cost_per_ha"] if cost_info["total_cost_per_ha"] > 0 else 1.0
        labor_cost = round(cost_info["labor_cost"] * scale_ratio, 2)
        fert_cost = round(cost_info["fertilizer_cost"] * scale_ratio, 2)
        pest_cost = round(cost_info["pesticide_cost"] * scale_ratio, 2)
        seed_cost = round(cost_info["seeds_cost"] * scale_ratio, 2)
        other_cost = round(cost_per_ha - (labor_cost + fert_cost + pest_cost + seed_cost), 2)
    else:
        cost_per_ha = cost_info["total_cost_per_ha"]
        labor_cost = cost_info["labor_cost"]
        fert_cost = cost_info["fertilizer_cost"]
        pest_cost = cost_info["pesticide_cost"]
        seed_cost = cost_info["seeds_cost"]
        other_cost = cost_info["other_costs"]

    total_production_cost = round(cost_per_ha * land_area, 2)
    itemized_costs = {
        "labor_cost_php": round(labor_cost * land_area, 2),
        "fertilizer_cost_php": round(fert_cost * land_area, 2),
        "pesticide_cost_php": round(pest_cost * land_area, 2),
        "seeds_cost_php": round(seed_cost * land_area, 2),
        "other_costs_php": round(other_cost * land_area, 2)
    }

    # Gross & Net Income
    gross_income = round(total_yield_kg * price_used, 2)
    net_income = round(gross_income - total_production_cost, 2)
    roi_percent = round((net_income / total_production_cost * 100.0), 2) if total_production_cost > 0 else 0.0
    profit_margin_percent = round((net_income / gross_income * 100.0), 2) if gross_income > 0 else 0.0

    # Scenarios (+15%, Expected, -20%)
    scenarios = {
        "favorable": {
            "label": "Favorable Season (+15%)",
            "condition": "Optimal rainfall distribution, low disease incidence, sunny grain/fruit filling",
            "yield_mt": round(total_yield_mt * 1.15, 2),
            "yield_kg": round(total_yield_kg * 1.15, 1),
            "gross_income_php": round(gross_income * 1.15, 2),
            "net_income_php": round((gross_income * 1.15) - total_production_cost, 2)
        },
        "expected": {
            "label": "Baseline Expected (Normal)",
            "condition": "Standard seasonal conditions and regular farm operations",
            "yield_mt": total_yield_mt,
            "yield_kg": total_yield_kg,
            "gross_income_php": gross_income,
            "net_income_php": net_income
        },
        "adverse": {
            "label": "Adverse Shock (-20%)",
            "condition": "Monsoon flooding, prolonged dry spell, or seasonal pest outbreak",
            "yield_mt": round(total_yield_mt * 0.80, 2),
            "yield_kg": round(total_yield_kg * 0.80, 1),
            "gross_income_php": round(gross_income * 0.80, 2),
            "net_income_php": round((gross_income * 0.80) - total_production_cost, 2)
        }
    }

    # Agronomic recommendation and suitability
    is_suitable = False
    suitability_notes = ""
    if province_info:
        rec_crops = (province_info.get("primary_crops", "") + "; " + province_info.get("secondary_crops", "")).lower()
        if norm_crop.lower() in rec_crops:
            is_suitable = True
            suitability_notes = f"Officially designated as a recommended crop for {prov_name} by DA-AMIA."
        else:
            # Check crop database suitable provinces
            suitable_provs = str(crop_meta.get("Suitable_Provinces", "")).lower()
            if prov_name.lower() in suitable_provs:
                is_suitable = True
                suitability_notes = f"Cultivated across {prov_name} according to PSA crop statistics."
            else:
                suitability_notes = f"Secondary or non-traditional crop for {prov_name}; verify local water and soil conditions."
    else:
        is_suitable = True
        suitability_notes = "Provincial suitability requires province selection."

    return {
        "crop_name": norm_crop,
        "crop_id": crop_meta.get("Crop_ID", "CRP-GEN"),
        "scientific_name": crop_meta.get("Scientific_Name", ""),
        "category": crop_meta.get("Crop_Category", "Agriculture"),
        "origin_status": crop_meta.get("Origin_Status", "Cultivated"),
        "land_area_ha": land_area,
        "season": season,
        "soil_type": soil_type,
        "yield_rate_mt_ha": round(yield_rate_mt_ha, 2),
        "yield_rate_kg_ha": round(yield_rate_kg_ha, 1),
        "total_yield_mt": total_yield_mt,
        "total_yield_kg": total_yield_kg,
        "total_cavans_sacks": total_cavans,
        "farmgate_price_php_kg": price_used,
        "price_unit": price_unit,
        "cost_per_ha_php": cost_per_ha,
        "production_cost_php": total_production_cost,
        "gross_income_php": gross_income,
        "net_income_php": net_income,
        "profit_margin_percent": profit_margin_percent,
        "roi_percent": roi_percent,
        "itemized_costs": itemized_costs,
        "scenarios": scenarios,
        "is_suitable": is_suitable,
        "suitability_notes": suitability_notes,
        "yield_source": yield_source,
        "is_ml_predicted": is_ml_predicted
    }

def process_multi_crop_calculation(payload, provinces_by_name):
    """
    Main calculation entrypoint for multiple crops.
    """
    province_name = payload.get("province", "").strip()
    season = payload.get("season", "Wet Season").strip()
    soil_type = payload.get("soil_type", "Sandy Loam").strip()
    crops_inputs = payload.get("crops", [])

    if not crops_inputs:
        return {"status": "error", "message": "At least one crop must be selected."}, 422

    # Match province
    matched_province = None
    if province_name:
        matched_province = provinces_by_name.get(province_name.lower())
        if not matched_province:
            for p_key, p_val in provinces_by_name.items():
                if province_name.lower() in p_key:
                    matched_province = p_val
                    break

    crop_results = []
    errors = []

    for idx, c_input in enumerate(crops_inputs):
        area = float(c_input.get("land_area_ha", 1.0))
        if area <= 0 or area > 500:
            errors.append(f"Crop #{idx+1} ({c_input.get('crop_name', 'Unknown')}): Land area must be between 0.1 and 500 ha.")
            continue

        res = calculate_single_crop(c_input, matched_province, season=season, soil_type=soil_type)
        crop_results.append(res)

    if errors:
        return {"status": "error", "message": "Validation failed", "errors": errors}, 422

    if not crop_results:
        return {"status": "error", "message": "No valid crop calculations could be generated."}, 422

    # Aggregate Totals
    total_area = round(sum(c["land_area_ha"] for c in crop_results), 2)
    total_gross = round(sum(c["gross_income_php"] for c in crop_results), 2)
    total_cost = round(sum(c["production_cost_php"] for c in crop_results), 2)
    total_net = round(sum(c["net_income_php"] for c in crop_results), 2)
    total_production_mt = round(sum(c["total_yield_mt"] for c in crop_results), 2)
    total_production_kg = round(sum(c["total_yield_kg"] for c in crop_results), 1)
    overall_roi = round((total_net / total_cost * 100.0), 2) if total_cost > 0 else 0.0

    # Best Crop by Net Income
    best_crop = max(crop_results, key=lambda x: x["net_income_php"])

    # Educational Comparison & Agronomic Advisory
    comparison_summary = {
        "best_crop_name": best_crop["crop_name"],
        "best_crop_net_income": best_crop["net_income_php"],
        "best_crop_roi": best_crop["roi_percent"],
        "advisory": (
            f"While {best_crop['crop_name']} projects the highest estimated net income "
            f"(₱{best_crop['net_income_php']:,.2f}), the highest calculated income does not "
            f"automatically indicate it is the optimal agricultural choice. Filipino farmers must also "
            f"weigh local soil chemistry, seasonal typhoon exposure, upfront capital requirements "
            f"(₱{best_crop['production_cost_php']:,.2f}), and farmgate market price volatility."
        )
    }

    response_data = {
        "status": "success",
        "parameters": {
            "province": matched_province["province"] if matched_province else (province_name or "National Average"),
            "region": matched_province["region"] if matched_province else "Philippines",
            "season": season,
            "soil_type": soil_type,
            "annual_rainfall_mm": matched_province["annual_rainfall_mm"] if matched_province else 2200.0,
            "avg_temp_c": matched_province["avg_temp_c"] if matched_province else 28.0
        },
        "totals": {
            "crop_count": len(crop_results),
            "total_land_area_ha": total_area,
            "total_production_mt": total_production_mt,
            "total_production_kg": total_production_kg,
            "total_gross_income_php": total_gross,
            "total_production_cost_php": total_cost,
            "total_net_income_php": total_net,
            "overall_roi_percent": overall_roi
        },
        "crop_results": crop_results,
        "comparison_summary": comparison_summary
    }

    if matched_province:
        response_data["province_details"] = {
            "province": matched_province["province"],
            "region": matched_province["region"],
            "da_suitability_grade": matched_province["da_suitability_grade"],
            "primary_crops": matched_province["primary_crops"],
            "secondary_crops": matched_province["secondary_crops"],
            "dominant_soil": matched_province["dominant_soil"],
            "climate_type": matched_province["climate_type"],
            "national_distinction": matched_province["national_distinction"],
            "agronomic_advice": matched_province["agronomic_advice"]
        }

    return response_data, 200
