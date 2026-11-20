"""
AniTaya System - Multi-Crop Agricultural Prediction & Financial Engine
Integrates:
- Leak-free Scikit-Learn ML Pipeline trained on real Philippine datasets
  (PSA OpenSTAT provincial yields + NASA POWER year-matched climate + PAGASA CliMap normals)
- Official PSA OpenSTAT Provincial Farmgate Price Series (DB/2M/NFG/0032M4AFN01..08.px)
- Official PSA OpenSTAT Table 5.5/5.6/5.7 Palay Costs & Returns (2022-2024) + DA reference costs
- Official PSA OpenSTAT Provincial Historical Crop Production Statistics (1987-2025)
- DA-AMIA Provincial Agro-Climatic Suitability Series (NCCAG v2.0)
"""

import json
import os
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "models", "anitaya_best_pipeline.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "models", "model_metrics.json")

# Official Philippine Datasets (with clean fallback check only if needed)
CROP_LIST_PATH = os.path.join(DATA_DIR, "philippine_crop_list.csv")
PSA_FG_PATH = os.path.join(DATA_DIR, "psa_provincial_farmgate_prices.csv")
PSA_PROD_PATH = os.path.join(DATA_DIR, "psa_provincial_production.csv")
PSA_PALAY_COST_PATH = os.path.join(DATA_DIR, "psa_palay_costs_returns.csv")

# In-memory cached state (single centralized instance shared with app.py)
crop_catalog = []
crop_by_name = {}
df_farmgate = pd.DataFrame()
df_prod = pd.DataFrame()
df_palay_cr = pd.DataFrame()
ml_pipeline = None
ml_metrics = {}
ml_supported_crops = set()
per_crop_metrics_map = {}

# Representative province fallback when a legacy /api/predict request sends region without province
REGION_REP_PROVINCE = {
    "Laguna / CALABARZON (Region IV-A)": "Laguna",
    "Central Luzon (Region III - Rice Granary)": "Nueva Ecija",
    "Ilocos Region (Region I)": "Ilocos Norte",
    "Cagayan Valley (Region II)": "Isabela",
    "Bicol Region (Region V)": "Camarines Sur",
    "Western Visayas (Region VI)": "Iloilo",
    "Davao Region (Region XI)": "Davao del Sur",
    "Northern Mindanao (Region X)": "Bukidnon",
    "Cordillera / Benguet Highlands": "Benguet",
}

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
    "goma": "Rubber",
}

CROP_SUITABILITY_ALIASES = {
    "rice, paddy": ("rice, paddy", "rice", "palay", "bigas"),
    "maize": ("maize", "corn", "mais", "yellow corn", "white corn"),
    "potatoes": ("potatoes", "potato", "patatas", "irish potato"),
    "cassava": ("cassava", "kamoteng kahoy", "balanghoy"),
}

# Reference Philippine Production Cost Estimates per hectare (PHP/ha) by crop category/crop.
# Palay (Rice, paddy) overrides dynamically from official PSA Table 5.5/5.6/5.7 CSV when loaded.
# All other crops carry transparent reference estimates and remain fully user-editable.
CATEGORY_COST_REFERENCE = {
    "Cereals & Grains": (38000.0, (0.35, 0.30, 0.08, 0.14, 0.13)),
    "Root & Tuber Crops": (36000.0, (0.42, 0.22, 0.06, 0.16, 0.14)),
    "Vegetables": (75000.0, (0.34, 0.25, 0.10, 0.18, 0.13)),
    "Legumes": (30000.0, (0.38, 0.24, 0.07, 0.18, 0.13)),
    "Fruits": (58000.0, (0.36, 0.28, 0.09, 0.13, 0.14)),
    "Commercial & Plantation": (55000.0, (0.40, 0.27, 0.07, 0.12, 0.14)),
}

CROP_COST_OVERRIDES = {
    "Potatoes": (95000.0, (0.24, 0.28, 0.10, 0.28, 0.10)),
    "Red Onion": (115000.0, (0.30, 0.22, 0.11, 0.27, 0.10)),
    "Yellow Onion": (115000.0, (0.30, 0.22, 0.11, 0.27, 0.10)),
    "Garlic": (105000.0, (0.28, 0.20, 0.09, 0.33, 0.10)),
    "Tomato": (82000.0, (0.35, 0.25, 0.11, 0.16, 0.13)),
    "Eggplant": (72000.0, (0.36, 0.25, 0.10, 0.15, 0.14)),
    "Cabbage": (78000.0, (0.34, 0.26, 0.12, 0.15, 0.13)),
    "Banana Cavendish": (135000.0, (0.34, 0.30, 0.12, 0.10, 0.14)),
    "Pineapple": (110000.0, (0.33, 0.28, 0.09, 0.18, 0.12)),
    "Sugarcane": (68000.0, (0.38, 0.29, 0.06, 0.15, 0.12)),
    "Coconut": (24000.0, (0.48, 0.22, 0.05, 0.10, 0.15)),
    "Tobacco": (65000.0, (0.40, 0.26, 0.08, 0.12, 0.14)),
    "Cassava": (34000.0, (0.44, 0.22, 0.06, 0.14, 0.14)),
    "Sweet potatoes": (29000.0, (0.44, 0.22, 0.06, 0.14, 0.14)),
}


def assess_crop_province_suitability(crop_name, province_info, crop_meta=None):
    """
    Unified suitability check combining:
      1) DA-AMIA provincial primary_crops / secondary_crops
      2) PSA crop catalog Suitable_Provinces (exact comma-token match)
    Returns (is_suitable: bool, is_da_recommended: bool, notes: str).
    """
    if not province_info:
        return True, False, "Provincial suitability requires province selection."

    prov_name = province_info.get("province", "").strip()
    prov_low = prov_name.lower()

    norm_crop = CROP_NAME_MAP.get(str(crop_name).strip().lower(), str(crop_name).strip())
    norm_low = norm_crop.lower()
    aliases = set(CROP_SUITABILITY_ALIASES.get(norm_low, (norm_low, str(crop_name).strip().lower())))

    primary_tokens = {c.strip().lower() for c in str(province_info.get("primary_crops", "")).split(";") if c.strip()}
    secondary_tokens = {c.strip().lower() for c in str(province_info.get("secondary_crops", "")).split(";") if c.strip()}
    in_primary = bool(aliases & primary_tokens)
    in_secondary = bool(aliases & secondary_tokens)
    is_da_recommended = in_primary or in_secondary

    if crop_meta is None:
        crop_meta = crop_by_name.get(norm_low, {})
    suitable_tokens = {p.strip().lower() for p in str(crop_meta.get("Suitable_Provinces", "")).split(",") if p.strip()}
    is_catalog_suitable = bool(prov_low and prov_low in suitable_tokens)

    if in_primary:
        return True, True, f"Officially designated as a primary crop for {prov_name} by DA-AMIA."
    if in_secondary:
        return True, True, f"Officially designated as a suitable rotational crop for {prov_name} by DA-AMIA."
    if is_catalog_suitable:
        return True, False, f"Major commercial producing province for {norm_crop} in PSA statistics."
    return False, False, f"Secondary or non-traditional crop for {prov_name}; verify local water and soil conditions."


def init_engine():
    global crop_catalog, crop_by_name, df_farmgate, df_prod, df_palay_cr
    global ml_pipeline, ml_metrics, ml_supported_crops, per_crop_metrics_map

    try:
        if os.path.exists(MODEL_PATH):
            ml_pipeline = joblib.load(MODEL_PATH)
            print("CropEngine: Centralized ML pipeline loaded successfully.")
    except Exception as e:
        print(f"CropEngine: Could not load ML pipeline: {e}")

    try:
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                ml_metrics = json.load(f)
            per_crop_metrics_map = ml_metrics.get("per_crop_metrics", {})
            if per_crop_metrics_map:
                ml_supported_crops = {
                    crop for crop, info in per_crop_metrics_map.items()
                    if info.get("ml_supported")
                }
            else:
                ml_supported_crops = set(ml_metrics.get("dataset_info", {}).get("crops", []))
    except Exception as e:
        print(f"CropEngine: Could not load ML metrics: {e}")

    try:
        if os.path.exists(CROP_LIST_PATH):
            df_c = pd.read_csv(CROP_LIST_PATH)
            crop_catalog = df_c.to_dict(orient="records")
            for c in crop_catalog:
                crop_by_name[c["Crop_Name"].strip().lower()] = c
            print(f"CropEngine: Loaded {len(crop_catalog)} Philippine crops.")
    except Exception as e:
        print(f"CropEngine: Error loading crop catalog: {e}")

    try:
        if os.path.exists(PSA_FG_PATH):
            df_farmgate = pd.read_csv(PSA_FG_PATH)
            print(f"CropEngine: Loaded {len(df_farmgate):,} official PSA farmgate prices.")
    except Exception as e:
        print(f"CropEngine: Error loading official PSA farmgate prices: {e}")

    try:
        if os.path.exists(PSA_PROD_PATH):
            df_prod = pd.read_csv(PSA_PROD_PATH)
            print(f"CropEngine: Loaded {len(df_prod):,} official PSA provincial production rows.")
    except Exception as e:
        print(f"CropEngine: Error loading official PSA crop production: {e}")

    try:
        if os.path.exists(PSA_PALAY_COST_PATH):
            df_palay_cr = pd.read_csv(PSA_PALAY_COST_PATH)
            print(f"CropEngine: Loaded {len(df_palay_cr)} official PSA Palay Costs & Returns rows.")
    except Exception as e:
        print(f"CropEngine: Error loading official PSA Palay Costs & Returns: {e}")

    # Enrich Crop Catalog with verified PSA prices, reference costs & empirical prediction tier
    try:
        for c in crop_catalog:
            c_name = c.get("Crop_Name", "")
            price_val, unit_str, price_src = get_farmgate_price_with_source(c_name)
            cost_info = get_production_cost(c_name)
            pm_meta = per_crop_metrics_map.get(c_name, {})
            c["Default_Price_PHP_kg"] = price_val
            c["Price_Unit"] = unit_str
            c["Price_Source"] = price_src
            c["Default_Cost_PHP_ha"] = cost_info["total_cost_per_ha"]
            c["Cost_Source"] = cost_info["cost_source"]
            c["Prediction_Method"] = pm_meta.get(
                "prediction_tier",
                "philippine_ml" if c_name in ml_supported_crops else "psa_statistical_historical",
            )
            c["ML_Supported"] = bool(c_name in ml_supported_crops)
    except Exception as e:
        print(f"CropEngine: Error enriching crop catalog: {e}")


def normalize_crop_name(raw_name):
    clean = raw_name.strip().lower()
    return CROP_NAME_MAP.get(clean, raw_name.strip())


def get_crop_info(crop_name):
    norm = normalize_crop_name(crop_name)
    info = crop_by_name.get(norm.lower())
    if not info:
        for k, v in crop_by_name.items():
            if norm.lower() in k or k in norm.lower():
                return v
    return info


def get_farmgate_price_with_source(crop_name, province_name=None):
    """
    Returns (price_php_per_kg, unit_label, source_description).
    Handles Sugarcane (mill-gate commercial cane reference instead of fresh chewing-cane retail)
    and Coconut (notes PSA per-nut farmgate parity vs kg-weight with husk cleanly).
    """
    norm = normalize_crop_name(crop_name)

    # Commercial Sugarcane harvested weight in PSA production is whole millable cane (MT/ha);
    # PSA NFG Table 08 tracks retail "Sugarcane for Chewing" (~PHP 11.9/kg) which does not match centrifugal mill-gate pricing.
    if norm == "Sugarcane":
        return 2.80, "PHP/kg (mill-gate cane ref)", "SRA / Commercial Mill-Gate Cane Reference (~PHP 2,800/MT; editable)"

    if not df_farmgate.empty:
        sub_crop = df_farmgate[df_farmgate["Crop_Name"].str.lower() == norm.lower()]
        if not sub_crop.empty:
            if province_name:
                match = sub_crop[sub_crop["Province"].str.lower() == province_name.strip().lower()]
                if not match.empty:
                    latest_yr = int(match["Year"].max())
                    row = match[match["Year"] == latest_yr].iloc[0]
                    raw_unit = str(row["Price_Unit"])
                    raw_src = str(row.get("Price_Source", "PSA OpenSTAT"))
                    if norm == "Coconut":
                        # PSA Coconut price is PHP/nut (avg mature nut ~1.2 kg with husk -> approx weight-basis parity note)
                        return round(float(row["Farmgate_Price"]), 2), "PHP/kg-equiv (from PHP/nut PSA)", f"{raw_src} ({latest_yr} Provincial Price; PHP/nut Basis)"
                    return round(float(row["Farmgate_Price"]), 2), raw_unit, f"{raw_src} ({latest_yr} Provincial Price)"

            # Fallback to median across provinces in latest available PSA year
            latest_yr = int(sub_crop["Year"].max())
            latest_sub = sub_crop[sub_crop["Year"] == latest_yr]
            med_price = round(float(latest_sub["Farmgate_Price"].median()), 2)
            raw_unit = str(latest_sub.iloc[0]["Price_Unit"])
            raw_src = str(latest_sub.iloc[0].get("Price_Source", "PSA OpenSTAT"))
            if norm == "Coconut":
                return med_price, "PHP/kg-equiv (from PHP/nut PSA)", f"{raw_src} ({latest_yr} National Median; PHP/nut Basis)"
            return med_price, raw_unit, f"{raw_src} ({latest_yr} National Median)"

    # Static fallback if dataset unavailable
    defaults = {
        "Rice, paddy": 17.5,
        "Maize": 13.5,
        "Potatoes": 38.0,
        "Cassava": 12.0,
        "Sweet potatoes": 17.5,
        "Plantains and others": 11.5,
        "Red Onion": 39.5,
        "Yellow Onion": 31.5,
        "Garlic": 98.0,
        "Tomato": 23.5,
        "Eggplant": 26.5,
    }
    return defaults.get(norm, 25.0), "PHP/kg", "Reference Estimate (No PSA Farmgate Match)"


def get_farmgate_price(crop_name, province_name=None):
    price, unit, _src = get_farmgate_price_with_source(crop_name, province_name)
    return price, unit


def get_production_cost(crop_name, province_name=None):
    """
    Returns itemized production cost breakdown in PHP/ha.
    Uses official PSA Table 5.5/5.6/5.7 Palay Costs & Returns for 'Rice, paddy';
    for all other crops uses transparently labeled DA/PSA reference cost estimates.
    """
    norm = normalize_crop_name(crop_name)

    if norm == "Rice, paddy" and not df_palay_cr.empty:
        all_palay = df_palay_cr[(df_palay_cr["Ecosystem"] == "All Palay") & (df_palay_cr["Semester"] == "Average")]
        if not all_palay.empty:
            max_yr = int(all_palay["Year"].max())
            yr_rows = all_palay[all_palay["Year"] == max_yr].set_index("Item")["Value"].to_dict()
            total_c = float(yr_rows.get("Total Costs", 59695.0))
            return {
                "total_cost_per_ha": round(total_c, 2),
                "labor_cost": round(total_c * 0.38, 2),
                "fertilizer_cost": round(total_c * 0.30, 2),
                "pesticide_cost": round(total_c * 0.08, 2),
                "seeds_cost": round(total_c * 0.10, 2),
                "other_costs": round(total_c * 0.14, 2),
                "cost_source": f"PSA OpenSTAT Table 5.5 All Palay Total Costs ({max_yr} Official National Avg; editable)",
                "is_official_psa_cost": True,
            }

    if norm in CROP_COST_OVERRIDES:
        total_c, shares = CROP_COST_OVERRIDES[norm]
    else:
        meta = get_crop_info(norm) or {}
        cat = meta.get("Crop_Category", "Cereals & Grains")
        total_c, shares = CATEGORY_COST_REFERENCE.get(cat, (45000.0, (0.38, 0.28, 0.08, 0.12, 0.14)))

    lab_s, fert_s, pest_s, seed_s, oth_s = shares
    return {
        "total_cost_per_ha": round(total_c, 2),
        "labor_cost": round(total_c * lab_s, 2),
        "fertilizer_cost": round(total_c * fert_s, 2),
        "pesticide_cost": round(total_c * pest_s, 2),
        "seeds_cost": round(total_c * seed_s, 2),
        "other_costs": round(total_c * oth_s, 2),
        "cost_source": "Reference Production Cost Estimate (Not an official PSA cost survey; editable)",
        "is_official_psa_cost": False,
    }


def get_historical_psa_yield(crop_name, province_name=None):
    """
    Look up verified recent official PSA provincial historical yield (MT/ha)
    from psa_provincial_production.csv.
    Returns (yield_mt_ha, scope_label, obs_count, latest_year).
    """
    norm = normalize_crop_name(crop_name)
    if df_prod.empty:
        return None, "Unavailable", 0, None

    crop_rows = df_prod[df_prod["item"].str.lower() == norm.lower()]
    if crop_rows.empty:
        return None, "Unavailable", 0, None

    # Prefer province-specific recent years (2020+)
    if province_name:
        prov_rows = crop_rows[crop_rows["area"].str.lower() == province_name.strip().lower()].sort_values("year")
        if not prov_rows.empty:
            recent = prov_rows.tail(4)
            med_y = round(float(recent["yield_mt_ha"].median()), 3)
            max_yr = int(recent["year"].max())
            min_yr = int(recent["year"].min())
            return med_y, f"PSA Official Historical Yield for {prov_rows.iloc[0]['area']} ({min_yr}-{max_yr} median)", len(prov_rows), max_yr

    # Fallback to recent national median across producing Philippine provinces
    recent_all = crop_rows[crop_rows["year"] >= max(2020, int(crop_rows["year"].max()) - 3)]
    med_nat = round(float(recent_all["yield_mt_ha"].median()), 3)
    max_yr = int(recent_all["year"].max())
    return (
        med_nat,
        f"PSA Official National Producing-Province Median ({int(recent_all['year'].min())}-{max_yr}; no PSA commercial record in selected province)",
        len(recent_all),
        max_yr,
    )


def get_baseline_yield_mt(crop_name, province_name=None):
    y_val, _lbl, _cnt, _yr = get_historical_psa_yield(crop_name, province_name)
    if y_val is not None:
        return y_val
    fallbacks = {
        "Rice, paddy": 3.5,
        "Maize": 2.8,
        "Cassava": 9.5,
        "Potatoes": 9.0,
        "Sweet potatoes": 6.3,
        "Plantains and others": 12.2,
        "Red Onion": 8.6,
        "Yellow Onion": 11.8,
        "Garlic": 2.9,
        "Tomato": 8.2,
        "Eggplant": 7.7,
    }
    return fallbacks.get(normalize_crop_name(crop_name), 4.0)


def get_province_psa_crop_summary(province_name):
    """Return dict of crop_name -> recent PSA provincial stats for UI / Advisor integration."""
    if df_prod.empty or not province_name:
        return {}
    prov_rows = df_prod[df_prod["area"].str.lower() == province_name.strip().lower()]
    if prov_rows.empty:
        return {}
    summary = {}
    for item_name, g in prov_rows.groupby("item"):
        recent = g.sort_values("year").tail(4)
        summary[item_name] = {
            "recent_yield_mt_ha": round(float(recent["yield_mt_ha"].median()), 2),
            "latest_year": int(recent["year"].max()),
            "latest_area_ha": round(float(recent.iloc[-1]["harvested_area_ha"]), 1),
            "latest_production_mt": round(float(recent.iloc[-1]["production_mt"]), 1),
            "observations_count": int(len(g)),
        }
    return summary


# Initialize engine at module import (single centralized instance)
init_engine()


def predict_crop_yield_rate(norm_crop, rainfall, avg_temp, season, prov_name, crop_meta, province_info=None):
    """
    Pure Crop Yield Prediction Rate (MT/ha) separated cleanly from financial math.
    Never uses hardcoded pesticides_tonnes=120.0 or Indonesia proxy!
    Uses explicit 3-tier classification:
      1. philippine_ml             -> Trained ML pipeline on verified PH province & climate
      2. psa_statistical_historical -> PSA official historical observation (sparse/low-R2/new crops)
      3. fallback_reference        -> Explicit fallback reference when neither exists
    Returns:
      (yield_rate_mt_ha, yield_source, prediction_method, method_label, is_ml_predicted)
    """
    best_info = ml_metrics.get("best_model", {}) if isinstance(ml_metrics, dict) else {}
    best_name = best_info.get("name", "Extra Trees Regressor")
    crop_eval = per_crop_metrics_map.get(norm_crop, {})

    target_prov = (prov_name or "").strip() or "Laguna"
    clamped_temp = max(14.0, min(34.0, float(avg_temp)))

    # Season-aware climate input using observed PAGASA seasonal ratios when available
    ann_rain = float(rainfall)
    if province_info and pd.notna(province_info.get("pagasa_wet_rain_mm")) and pd.notna(province_info.get("pagasa_dry_rain_mm")):
        wet_r = float(province_info["pagasa_wet_rain_mm"])
        dry_r = float(province_info["pagasa_dry_rain_mm"])
        if "wet" in season.lower():
            # Reflect fuller wet-season water availability relative to province climate baseline
            effective_rainfall = max(ann_rain, wet_r * 1.15)
        else:
            effective_rainfall = min(ann_rain, max(900.0, dry_r * 1.85))
    else:
        effective_rainfall = ann_rain if "wet" in season.lower() else min(ann_rain, ann_rain * 0.78)
    effective_rainfall = max(400.0, min(5500.0, effective_rainfall))

    if norm_crop in ml_supported_crops and ml_pipeline is not None:
        input_df = pd.DataFrame([{
            "item": norm_crop,
            "average_rain_fall_mm_per_year": round(effective_rainfall, 1),
            "avg_temp": round(clamped_temp, 2),
            "area": target_prov,
        }])
        predicted_hg_ha = float(ml_pipeline.predict(input_df)[0])
        predicted_hg_ha = max(predicted_hg_ha, 200.0)  # Physical lower floor 0.02 MT/ha (preserves Coffee/Abaca scale)
        yield_rate_mt_ha = predicted_hg_ha / 10000.0

        # Season-specific agronomic adjustment for lowland rice/corn where Dry irrigated outyields wet typhoon season slightly
        if norm_crop == "Rice, paddy" and "dry" in season.lower():
            yield_rate_mt_ha *= 1.04
        elif norm_crop == "Maize" and "dry" in season.lower():
            yield_rate_mt_ha *= 1.03

        c_r2 = crop_eval.get("test_r2")
        r2_note = f"Crop Holdout R²={c_r2:.2f}" if isinstance(c_r2, (int, float)) else f"Overall Test R²={best_info.get('test_r2', 0):.2f}"
        return (
            round(yield_rate_mt_ha, 4),
            f"{best_name} trained on Philippine PSA + NASA POWER/PAGASA ({target_prov}, {r2_note})",
            "philippine_ml",
            "Philippine ML Model",
            True,
        )

    # Tier 2: Verified PSA Provincial Historical Observations
    hist_yield, hist_label, obs_cnt, _yr = get_historical_psa_yield(norm_crop, target_prov)
    if hist_yield is not None and hist_yield > 0:
        if "wet" in season.lower() and norm_crop in ("Red Onion", "Yellow Onion", "Garlic", "Potatoes", "Tomato", "Watermelon"):
            # Dry-season bulb/tuber/solanaceous crops face moisture blight stress during peak monsoon
            season_factor = 0.88
            hist_label += " (adjusted -12% for Wet Season disease/moisture risk)"
        else:
            season_factor = 1.0
        return (
            round(hist_yield * season_factor, 4),
            hist_label,
            "psa_statistical_historical",
            "PSA Historical Stats",
            False,
        )

    # Tier 3: Reference Baseline Fallback
    base_mt = get_baseline_yield_mt(norm_crop, target_prov)
    return (
        round(base_mt, 4),
        "Reference Agronomic Baseline Estimate (Insufficient PSA provincial data for ML)",
        "fallback_reference",
        "Reference Fallback",
        False,
    )


def calculate_single_crop(crop_input, province_info, season="Wet Season", soil_type="Loam"):
    """
    Computes yield (via predict_crop_yield_rate) and separate financial projections for one crop.
    Note: soil_type is preserved solely as informational agronomic metadata and is NOT an ML input feature.
    """
    raw_crop = crop_input.get("crop_name", "Rice, paddy")
    norm_crop = normalize_crop_name(raw_crop)
    land_area = float(crop_input.get("land_area_ha", 1.0))

    crop_meta = get_crop_info(norm_crop) or {
        "Crop_Name": norm_crop,
        "Scientific_Name": "Agricultural crop",
        "Crop_Category": "General Agriculture",
        "Origin_Status": "Cultivated",
        "Description": "Cultivated Philippine crop.",
    }

    prov_name = province_info.get("province", "") if province_info else ""
    if not prov_name:
        prov_name = "Laguna"
    rainfall = float(province_info.get("annual_rainfall_mm", 2200.0)) if province_info else 2200.0
    avg_temp = float(province_info.get("avg_temp_c", 27.5)) if province_info else 27.5

    # Step 1: Pure Crop Yield Prediction (separated from financials)
    yield_rate_mt_ha, yield_source, pred_method, method_label, is_ml_predicted = predict_crop_yield_rate(
        norm_crop=norm_crop,
        rainfall=rainfall,
        avg_temp=avg_temp,
        season=season,
        prov_name=prov_name,
        crop_meta=crop_meta,
        province_info=province_info,
    )
    yield_rate_kg_ha = round(yield_rate_mt_ha * 1000.0, 1)
    total_yield_mt = round(yield_rate_mt_ha * land_area, 3)
    total_yield_kg = round(total_yield_mt * 1000.0, 1)

    cavan_weight = 50.0
    total_cavans = round(total_yield_kg / cavan_weight, 1)

    # Step 2: Farmgate Price
    custom_price = crop_input.get("farmgate_price")
    if custom_price is not None and str(custom_price).strip() != "" and float(custom_price) > 0:
        price_used = round(float(custom_price), 2)
        price_unit = "PHP/kg (Custom User Input)"
        price_source = "User-Specified Custom Farmgate Price"
    else:
        price_used, price_unit, price_source = get_farmgate_price_with_source(norm_crop, prov_name)
    price_per_mt = round(price_used * 1000.0, 2)

    # Step 3: Production Cost (Editable custom cost supported)
    custom_cost = crop_input.get("cost_per_ha")
    cost_info = get_production_cost(norm_crop, prov_name)
    if custom_cost is not None and str(custom_cost).strip() != "" and float(custom_cost) > 0:
        cost_per_ha = round(float(custom_cost), 2)
        scale_ratio = cost_per_ha / cost_info["total_cost_per_ha"] if cost_info["total_cost_per_ha"] > 0 else 1.0
        labor_cost = round(cost_info["labor_cost"] * scale_ratio, 2)
        fert_cost = round(cost_info["fertilizer_cost"] * scale_ratio, 2)
        pest_cost = round(cost_info["pesticide_cost"] * scale_ratio, 2)
        seed_cost = round(cost_info["seeds_cost"] * scale_ratio, 2)
        other_cost = round(cost_per_ha - (labor_cost + fert_cost + pest_cost + seed_cost), 2)
        cost_source = "User-Specified Custom Cost/ha"
        is_official_cost = False
    else:
        cost_per_ha = cost_info["total_cost_per_ha"]
        labor_cost = cost_info["labor_cost"]
        fert_cost = cost_info["fertilizer_cost"]
        pest_cost = cost_info["pesticide_cost"]
        seed_cost = cost_info["seeds_cost"]
        other_cost = cost_info["other_costs"]
        cost_source = cost_info["cost_source"]
        is_official_cost = cost_info["is_official_psa_cost"]

    total_production_cost = round(cost_per_ha * land_area, 2)
    itemized_costs = {
        "labor_cost_php": round(labor_cost * land_area, 2),
        "fertilizer_cost_php": round(fert_cost * land_area, 2),
        "pesticide_cost_php": round(pest_cost * land_area, 2),
        "seeds_cost_php": round(seed_cost * land_area, 2),
        "other_costs_php": round(other_cost * land_area, 2),
    }

    # Step 4: Revenue & Net Income Estimates
    gross_income = round(total_yield_mt * price_per_mt, 2)
    net_income = round(gross_income - total_production_cost, 2)
    roi_percent = round((net_income / total_production_cost * 100.0), 2) if total_production_cost > 0 else 0.0
    profit_margin_percent = round((net_income / gross_income * 100.0), 2) if gross_income > 0 else 0.0

    scenarios = {
        "favorable": {
            "label": "Favorable Season (+15%)",
            "condition": "Optimal rainfall distribution, low disease pressure, favorable harvest window",
            "yield_mt": round(total_yield_mt * 1.15, 2),
            "yield_kg": round(total_yield_kg * 1.15, 1),
            "gross_income_php": round(gross_income * 1.15, 2),
            "net_income_php": round((gross_income * 1.15) - total_production_cost, 2),
        },
        "expected": {
            "label": "Baseline Expected Estimate (Normal)",
            "condition": "Typical provincial seasonal climate and normal farm input management",
            "yield_mt": total_yield_mt,
            "yield_kg": total_yield_kg,
            "gross_income_php": gross_income,
            "net_income_php": net_income,
        },
        "adverse": {
            "label": "Adverse Shock (-20%)",
            "condition": "Typhoon/monsoon flood stress, El Niño dry spell, or pest incidence",
            "yield_mt": round(total_yield_mt * 0.80, 2),
            "yield_kg": round(total_yield_kg * 0.80, 1),
            "gross_income_php": round(gross_income * 0.80, 2),
            "net_income_php": round((gross_income * 0.80) - total_production_cost, 2),
        },
    }

    is_suitable, _is_da_rec, suitability_notes = assess_crop_province_suitability(norm_crop, province_info, crop_meta)
    crop_eval = per_crop_metrics_map.get(norm_crop, {})

    return {
        "crop_name": norm_crop,
        "crop_id": crop_meta.get("Crop_ID", "CRP-GEN"),
        "scientific_name": crop_meta.get("Scientific_Name", ""),
        "category": crop_meta.get("Crop_Category", "Agriculture"),
        "origin_status": crop_meta.get("Origin_Status", "Cultivated"),
        "land_area_ha": land_area,
        "season": season,
        "soil_type": soil_type,
        "soil_is_ml_feature": False,
        "yield_rate_mt_ha": round(yield_rate_mt_ha, 2),
        "yield_rate_kg_ha": round(yield_rate_kg_ha, 1),
        "total_yield_mt": total_yield_mt,
        "total_yield_kg": total_yield_kg,
        "total_cavans_sacks": total_cavans,
        "farmgate_price_php_kg": price_used,
        "farmgate_price_php_mt": price_per_mt,
        "price_unit": price_unit,
        "price_source": price_source,
        "cost_per_ha_php": cost_per_ha,
        "cost_source": cost_source,
        "is_official_psa_cost": is_official_cost,
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
        "prediction_method": pred_method,
        "prediction_method_label": method_label,
        "crop_test_r2": crop_eval.get("test_r2"),
        "crop_test_mae_mt_ha": crop_eval.get("test_mae_mt_ha"),
        "is_ml_predicted": is_ml_predicted,
    }


def process_multi_crop_calculation(payload, provinces_by_name):
    """Main calculation entrypoint for multiple crops."""
    province_name = payload.get("province", "").strip()
    season = payload.get("season", "Wet Season").strip()
    soil_type = payload.get("soil_type", "Loam").strip()
    crops_inputs = payload.get("crops", [])

    if not crops_inputs:
        return {"status": "error", "message": "At least one crop must be selected."}, 422

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

    total_area = round(sum(c["land_area_ha"] for c in crop_results), 2)
    total_gross = round(sum(c["gross_income_php"] for c in crop_results), 2)
    total_cost = round(sum(c["production_cost_php"] for c in crop_results), 2)
    total_net = round(sum(c["net_income_php"] for c in crop_results), 2)
    total_production_mt = round(sum(c["total_yield_mt"] for c in crop_results), 2)
    total_production_kg = round(sum(c["total_yield_kg"] for c in crop_results), 1)
    overall_roi = round((total_net / total_cost * 100.0), 2) if total_cost > 0 else 0.0

    best_crop = max(crop_results, key=lambda x: x["net_income_php"])

    comparison_summary = {
        "best_crop_name": best_crop["crop_name"],
        "best_crop_net_income": best_crop["net_income_php"],
        "best_crop_roi": best_crop["roi_percent"],
        "advisory": (
            f"While {best_crop['crop_name']} projects the highest estimated net income "
            f"(₱{best_crop['net_income_php']:,.2f}), the highest calculated income does not "
            f"automatically indicate it is the optimal agricultural choice. Filipino farmers must also "
            f"weigh local soil chemistry, seasonal typhoon exposure, upfront capital requirements "
            f"(₱{best_crop['production_cost_php']:,.2f}), and farmgate market price volatility. "
            f"All yield and peso figures are planning estimates, not financial guarantees."
        ),
    }

    best_info = ml_metrics.get("best_model", {}) if isinstance(ml_metrics, dict) else {}
    response_data = {
        "status": "success",
        "parameters": {
            "province": matched_province["province"] if matched_province else (province_name or "Laguna (Fallback)"),
            "region": matched_province["region"] if matched_province else "Philippines",
            "season": season,
            "soil_type": soil_type,
            "soil_note": "Informational agronomic metadata only (not an input feature to the ML model).",
            "annual_rainfall_mm": matched_province["annual_rainfall_mm"] if matched_province else 2200.0,
            "avg_temp_c": matched_province["avg_temp_c"] if matched_province else 27.5,
            "climate_data_source": matched_province.get("climate_data_source", "DOST-PAGASA CliMap v2.1 Normals (1991-2020)") if matched_province else "Philippine Reference Average",
        },
        "model_summary": {
            "model_name": best_info.get("name", "Extra Trees Regressor"),
            "cv_r2_mean": best_info.get("cv_r2_mean"),
            "test_r2": best_info.get("test_r2"),
            "test_mae_mt_ha": round(best_info.get("test_mae", 0) / 10000.0, 2) if best_info.get("test_mae") else None,
        },
        "totals": {
            "crop_count": len(crop_results),
            "total_land_area_ha": total_area,
            "total_production_mt": total_production_mt,
            "total_production_kg": total_production_kg,
            "total_gross_income_php": total_gross,
            "total_production_cost_php": total_cost,
            "total_net_income_php": total_net,
            "overall_roi_percent": overall_roi,
        },
        "crop_results": crop_results,
        "comparison_summary": comparison_summary,
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
            "agronomic_advice": matched_province["agronomic_advice"],
        }

    return response_data, 200
