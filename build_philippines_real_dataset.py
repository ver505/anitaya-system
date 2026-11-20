"""
AniTaya System - Real Philippine Agricultural & Climate Dataset Integration Pipeline
===================================================================================
Replaces the global Indonesia proxy and synthetic provincial hash tables with verified
Philippine observations from official public APIs:

1. Production & Harvested Area (PSA OpenSTAT PxWeb REST API - DB/2E/CS):
   - Palay and Corn: 0012E4EVCP0.px & 0022E4EAHC0.px (1987-2025, Annual, by Province)
   - Vegetables & Root Crops: 0082E4EVCP3.px & 0112E4EAHM3.px (2010-2025, Annual, by Province)
   - Fruit Crops: 0072E4EVCP2.px & 0102E4EAHM2.px (2010-2025, Annual, by Province)
   - Non-Food & Industrial Crops: 0072E4EVCP1.px & 0092E4EAHM1.px (2010-2025, Annual, by Province)
   Source URL: https://openstat.psa.gov.ph/PXWeb/api/v1/en/DB/2E/CS

2. Official Farmgate Prices & Palay Costs/Returns (PSA OpenSTAT PxWeb REST API):
   - Farmgate Prices (New Series): DB/2M/NFG/0032M4AFN01.px .. 0032M4AFN08.px (2010-2025)
   - Table 5.5/5.6/5.7 Costs & Returns of Palay Production: DB/3S/C5/0052E2BARP0.px .. 0072E2BARN0.px (2022-2024)

3. Official PAGASA DOST CliMap v2.1 Observed Gridded Climate Normals (1991-2020):
   - Province boundaries: https://climgridph.pagasa.dost.gov.ph/climapv2/assets/js/PH_Adm2_Provinces.json
   - Observed monthly rainfall rasters: maps/rr/tif/01.tif .. 12.tif (mm)
   - Observed monthly mean temperature rasters: maps/tm/tif/01.tif .. 12.tif (deg C)
   (Note: ClimGridPh's PHP station-history endpoint APIgetHSProvince.php and PhilRice PalayStat
    timed out during automated retrieval and are supported for optional manual drop-in CSVs
    under data/manual_imports/.)

4. Year-Matched Historical Weather Time Series (1987-2025) per Province Centroid:
   - NASA POWER Agroclimatology Monthly API (MERRA-2 Reanalysis / Satellite Precipitation):
     https://power.larc.nasa.gov/api/temporal/monthly/point
   - Matched to each province's official PAGASA polygon centroid (lat, lon) and exact year.
   - Clearly labeled in climate_source so NASA POWER is never mislabeled as PAGASA station data.
"""

import calendar
import csv
import io
import json
import math
import os
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import requests
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MANUAL_DIR = os.path.join(DATA_DIR, "manual_imports")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MANUAL_DIR, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (AniTaya-Agricultural-Pipeline/2.0)"}
PSA_BASE = "https://openstat.psa.gov.ph/PXWeb/api/v1/en"
PAGASA_BASE = "https://climgridph.pagasa.dost.gov.ph/climapv2"
NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"

# Province name reconciliation from PSA OpenSTAT labels -> AniTaya 82 canonical provinces
PSA_PROVINCE_ALIASES = {
    "davao de oro (compostela valley)": "Davao de Oro",
    "compostela valley": "Davao de Oro",
    "cotabato (north cotabato)": "Cotabato",
    "north cotabato": "Cotabato",
    "tawi-tawi": "Tawi-Tawi",
    "samar (western samar)": "Samar",
    "western samar": "Samar",
    "maguindanao": "Maguindanao",
    "maguindanao del norte": "Maguindanao",
    "maguindanao del sur": "Maguindanao",
}


def log(msg):
    print(msg, flush=True)


def norm_text(s):
    s = unicodedata.normalize("NFKD", str(s or "")).strip()
    s = s.lstrip(". ").strip()
    for suf in (" a/", " b/", " c/", " d/", " r/"):
        if s.lower().endswith(suf):
            s = s[: -len(suf)].strip()
    return s


def load_canonical_provinces():
    path = os.path.join(DATA_DIR, "philippine_provinces_crop_suitability.csv")
    df = pd.read_csv(path)
    canon = {}
    for _, row in df.iterrows():
        p = str(row["province"]).strip()
        canon[p.lower()] = {
            "province": p,
            "region": str(row["region"]).strip(),
        }
    return df, canon


def resolve_psa_province(raw_geo, canon_map):
    cleaned = norm_text(raw_geo)
    low = cleaned.lower()
    if low in PSA_PROVINCE_ALIASES:
        mapped = PSA_PROVINCE_ALIASES[low]
        if mapped.lower() in canon_map:
            return mapped
    if low in canon_map:
        return canon_map[low]["province"]
    return None


def safe_float(val):
    if val is None:
        return None
    s = str(val).strip()
    if s in ("", "..", "...", "-", "n/a", "N/A", "nan", "NaN", "c", "p"):
        return None
    try:
        v = float(s.replace(",", ""))
        return v if math.isfinite(v) else None
    except Exception:
        return None


def http_get_with_retry(url, timeout=45, retries=5):
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            if r.status_code == 429:
                retry_after = safe_float(r.headers.get("Retry-After"))
                wait_s = max(62.0, retry_after + 2.0) if retry_after else 62.0
                log(f"  Rate-limited (HTTP 429). Waiting {int(wait_s)}s before retry {attempt + 1}/{retries}...")
                time.sleep(wait_s)
                continue
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            time.sleep(2.5 * (attempt + 1))
    raise RuntimeError(f"GET failed for {url}: {last_err}")


def http_post_json_with_retry(url, payload, timeout=60, retries=3):
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.post(url, json=payload, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"POST failed for {url}: {last_err}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Official PAGASA CliMap v2.1 Gridded Climate Normals (1991-2020)
# ─────────────────────────────────────────────────────────────────────────────
def poly_bbox(geom):
    pts = []
    if geom["type"] == "Polygon":
        polys = geom["coordinates"]
    elif geom["type"] == "MultiPolygon":
        polys = [ring for poly in geom["coordinates"] for ring in poly]
    else:
        polys = []
    for ring in polys:
        pts.extend(ring)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def point_in_geom(lon, lat, geom):
    def in_ring(x, y, ring):
        inside = False
        n = len(ring)
        j = n - 1
        for i in range(n):
            xi, yi = ring[i][0], ring[i][1]
            xj, yj = ring[j][0], ring[j][1]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-15) + xi):
                inside = not inside
            j = i
        return inside

    if geom["type"] == "Polygon":
        return in_ring(lon, lat, geom["coordinates"][0])
    if geom["type"] == "MultiPolygon":
        for poly in geom["coordinates"]:
            if in_ring(lon, lat, poly[0]):
                return True
    return False


def lonlat_to_mercator(lon, lat):
    radius = 6378137.0
    x = radius * math.radians(lon)
    y = radius * math.log(math.tan(math.pi / 4.0 + math.radians(lat) / 2.0))
    return x, y


def fetch_pagasa_climap_normals(canon_map):
    out_path = os.path.join(DATA_DIR, "pagasa_climap_province_normals.csv")
    if os.path.exists(out_path):
        try:
            existing = pd.read_csv(out_path)
            if len(existing) >= len(canon_map) and existing["pagasa_normal_rain_mm"].notna().all():
                log(f"\n[1/5] Loaded cached PAGASA CliMap normals ({len(existing)} provinces) from {out_path}")
                return existing
        except Exception:
            pass

    log("\n[1/5] Fetching PAGASA DOST CliMap v2.1 province boundaries & monthly observed GeoTIFFs...")
    geo_url = f"{PAGASA_BASE}/assets/js/PH_Adm2_Provinces.json"
    gj = http_get_with_retry(geo_url, timeout=45).json()

    rr_arrays = {}
    tm_arrays = {}
    geo_transform = None

    for m in range(1, 13):
        mm = f"{m:02d}"
        rr_resp = http_get_with_retry(f"{PAGASA_BASE}/maps/rr/tif/{mm}.tif", timeout=45)
        tm_resp = http_get_with_retry(f"{PAGASA_BASE}/maps/tm/tif/{mm}.tif", timeout=45)
        im_rr = Image.open(io.BytesIO(rr_resp.content))
        im_tm = Image.open(io.BytesIO(tm_resp.content))
        if geo_transform is None:
            tie = im_rr.tag_v2[33922]
            scale = im_rr.tag_v2[33550]
            geo_transform = (float(tie[3]), float(tie[4]), float(scale[0]), float(scale[1]))
        rr_arrays[m] = np.array(im_rr)
        tm_arrays[m] = np.array(im_tm)
        log(f"  Loaded PAGASA CliMap month {mm}/12 (rr + tm rasters)")

    x0, y0, sx, sy = geo_transform
    h, w = rr_arrays[1].shape

    # Collect pixel coordinates per canonical AniTaya province
    prov_pixels = {}
    for feat in gj.get("features", []):
        adm2 = str(feat.get("properties", {}).get("ADM2_EN", "")).strip()
        if not adm2 or "not a province" in adm2.lower():
            continue
        p_canon = None
        if adm2.lower() in canon_map:
            p_canon = canon_map[adm2.lower()]["province"]
        elif adm2.lower() in PSA_PROVINCE_ALIASES and PSA_PROVINCE_ALIASES[adm2.lower()].lower() in canon_map:
            p_canon = PSA_PROVINCE_ALIASES[adm2.lower()]
        if not p_canon:
            continue

        geom = feat["geometry"]
        minx, miny, maxx, maxy = poly_bbox(geom)
        sample_coords = []
        # Dense 14x14 spatial grid per province bbox to average interior pixels accurately
        for glon in np.linspace(minx, maxx, 14):
            for glat in np.linspace(miny, maxy, 14):
                if point_in_geom(float(glon), float(glat), geom):
                    mx, my = lonlat_to_mercator(float(glon), float(glat))
                    col = int(round((mx - x0) / sx))
                    row = int(round((y0 - my) / sy))
                    if 0 <= row < h and 0 <= col < w:
                        sample_coords.append((row, col, float(glat), float(glon)))

        if not sample_coords:
            clon, clat = (minx + maxx) / 2.0, (miny + maxy) / 2.0
            mx, my = lonlat_to_mercator(clon, clat)
            col = min(max(0, int(round((mx - x0) / sx))), w - 1)
            row = min(max(0, int(round((y0 - my) / sy))), h - 1)
            sample_coords.append((row, col, clat, clon))

        prov_pixels[p_canon] = sample_coords

    # Explicit coordinate fallbacks for Metro Manila / any micro-island edge cases
    records = []
    for p_low, p_info in sorted(canon_map.items()):
        p_name = p_info["province"]
        coords = prov_pixels.get(p_name)
        if not coords:
            # Handle Metro Manila / tiny islands if filtered from Adm2 ("NCR, City of Manila..." in ADM2)
            # Use NCR districts if Metro Manila
            if p_name.lower() == "metro manila":
                ncr_pts = []
                for feat in gj.get("features", []):
                    if "ncr" in str(feat.get("properties", {}).get("ADM2_EN", "")).lower():
                        minx, miny, maxx, maxy = poly_bbox(feat["geometry"])
                        for glon in np.linspace(minx, maxx, 8):
                            for glat in np.linspace(miny, maxy, 8):
                                if point_in_geom(float(glon), float(glat), feat["geometry"]):
                                    mx, my = lonlat_to_mercator(float(glon), float(glat))
                                    col = int(round((mx - x0) / sx))
                                    row = int(round((y0 - my) / sy))
                                    if 0 <= row < h and 0 <= col < w:
                                        ncr_pts.append((row, col, float(glat), float(glon)))
                coords = ncr_pts if ncr_pts else [(0, 0, 14.5995, 120.9842)]
            else:
                continue

        c_lat = float(np.mean([pt[2] for pt in coords]))
        c_lon = float(np.mean([pt[3] for pt in coords]))

        row_dict = {
            "province": p_name,
            "region": p_info["region"],
            "centroid_lat": round(c_lat, 5),
            "centroid_lon": round(c_lon, 5),
            "sample_pixels": len(coords),
        }

        monthly_rr = []
        monthly_tm = []
        for m in range(1, 13):
            rvals = []
            tvals = []
            for r_idx, c_idx, _, _ in coords:
                rv = float(rr_arrays[m][r_idx, c_idx])
                tv = float(tm_arrays[m][r_idx, c_idx])
                if rv > -900 and math.isfinite(rv):
                    rvals.append(rv)
                if tv > -900 and math.isfinite(tv):
                    tvals.append(tv)
            # If coastal boundary pixel had -9999 nodata, widen 3-pixel local window
            if not rvals or not tvals:
                r0, c0 = coords[len(coords) // 2][0], coords[len(coords) // 2][1]
                for dr in range(-4, 5):
                    for dc in range(-4, 5):
                        rr2 = min(max(0, r0 + dr), h - 1)
                        cc2 = min(max(0, c0 + dc), w - 1)
                        rv = float(rr_arrays[m][rr2, cc2])
                        tv = float(tm_arrays[m][rr2, cc2])
                        if rv > -900 and math.isfinite(rv):
                            rvals.append(rv)
                        if tv > -900 and math.isfinite(tv):
                            tvals.append(tv)
            m_rain = round(float(np.mean(rvals)), 2) if rvals else np.nan
            m_temp = round(float(np.mean(tvals)), 2) if tvals else np.nan
            row_dict[f"rr_{m:02d}_mm"] = m_rain
            row_dict[f"tm_{m:02d}_c"] = m_temp
            if not math.isnan(m_rain):
                monthly_rr.append(m_rain)
            if not math.isnan(m_temp):
                monthly_tm.append(m_temp)

        row_dict["pagasa_normal_rain_mm"] = round(float(np.sum(monthly_rr)), 1) if len(monthly_rr) == 12 else np.nan
        row_dict["pagasa_normal_temp_c"] = round(float(np.mean(monthly_tm)), 2) if len(monthly_tm) == 12 else np.nan
        row_dict["pagasa_wet_rain_mm"] = round(sum(row_dict[f"rr_{m:02d}_mm"] for m in [6, 7, 8, 9, 10, 11]), 1)
        row_dict["pagasa_dry_rain_mm"] = round(sum(row_dict[f"rr_{m:02d}_mm"] for m in [12, 1, 2, 3, 4, 5]), 1)
        row_dict["source"] = "DOST-PAGASA CliMap v2.1 ClimGridPh Observed GeoTIFF Normals (1991-2020)"
        records.append(row_dict)

    df_pag = pd.DataFrame(records)
    out_path = os.path.join(DATA_DIR, "pagasa_climap_province_normals.csv")
    df_pag.to_csv(out_path, index=False)
    log(f"  Saved {len(df_pag)} provinces PAGASA CliMap normals -> {out_path}")
    return df_pag


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Year-Matched Historical Weather (1987-2025) from NASA POWER (MERRA-2)
# ─────────────────────────────────────────────────────────────────────────────
def _fetch_single_province_nasa_power(p_info, start_year, end_year):
    prov = p_info["province"]
    lat = p_info["centroid_lat"]
    lon = p_info["centroid_lon"]
    url = (
        f"{NASA_POWER_URL}?parameters=T2M,PRECTOTCORR&community=AG"
        f"&longitude={lon}&latitude={lat}&start={start_year}&end={end_year}&format=JSON"
    )
    r = http_get_with_retry(url, timeout=60, retries=5)
    params = r.json().get("properties", {}).get("parameter", {})
    t2m = params.get("T2M", {})
    prec = params.get("PRECTOTCORR", {})

    rows = []
    for yr in range(start_year, end_year + 1):
        temps = []
        wet_temps, dry_temps = [], []
        wet_rain, dry_rain = 0.0, 0.0
        total_rain = 0.0
        valid_months = 0
        for mo in range(1, 13):
            key = f"{yr}{mo:02d}"
            tv = t2m.get(key)
            pv = prec.get(key)
            if tv is None or pv is None or float(tv) < -900 or float(pv) < -900:
                continue
            valid_months += 1
            days_in_mo = calendar.monthrange(yr, mo)[1]
            mo_rain_mm = float(pv) * days_in_mo
            mo_temp_c = float(tv)
            total_rain += mo_rain_mm
            temps.append(mo_temp_c)
            if mo in (6, 7, 8, 9, 10, 11):
                wet_rain += mo_rain_mm
                wet_temps.append(mo_temp_c)
            else:
                dry_rain += mo_rain_mm
                dry_temps.append(mo_temp_c)
        if valid_months < 12:
            continue
        rows.append({
            "area": prov,
            "year": int(yr),
            "average_rain_fall_mm_per_year": round(total_rain, 1),
            "avg_temp": round(float(np.mean(temps)), 2),
            "wet_season_rain_mm": round(wet_rain, 1),
            "dry_season_rain_mm": round(dry_rain, 1),
            "wet_season_temp_c": round(float(np.mean(wet_temps)), 2),
            "dry_season_temp_c": round(float(np.mean(dry_temps)), 2),
            "climate_source": "NASA POWER Agroclimatology Monthly API (MERRA-2, Province Centroid Yearly Matched)",
        })
    return prov, rows


def fetch_yearly_historical_climate(df_pag, start_year=1987, end_year=2025):
    log(f"\n[2/5] Fetching year-matched monthly->annual historical climate ({start_year}-{end_year}) via NASA POWER Agroclimatology API...")
    out_path = os.path.join(DATA_DIR, "philippines_province_yearly_climate.csv")

    all_rows = []
    done_provs = set()
    if os.path.exists(out_path):
        try:
            df_existing = pd.read_csv(out_path)
            if len(df_existing) > 0 and "climate_source" in df_existing.columns:
                df_nasa = df_existing[df_existing["climate_source"].astype(str).str.contains("NASA POWER", na=False)]
                if len(df_nasa) > 0:
                    all_rows = df_nasa.to_dict(orient="records")
                    done_provs = set(df_nasa["area"].dropna().unique())
                    log(f"  Resuming NASA POWER checkpoint: {len(done_provs)} / {len(df_pag)} provinces complete.")
        except Exception:
            all_rows = []
            done_provs = set()

    provs = [
        p for p in df_pag[["province", "centroid_lat", "centroid_lon"]].to_dict(orient="records")
        if p["province"] not in done_provs
    ]
    if provs:
        completed = 0
        with ThreadPoolExecutor(max_workers=2) as pool:
            futs = [pool.submit(_fetch_single_province_nasa_power, p, start_year, end_year) for p in provs]
            for fut in as_completed(futs):
                prov, rows = fut.result()
                all_rows.extend(rows)
                completed += 1
                if completed % 10 == 0 or completed == len(provs):
                    pd.DataFrame(all_rows).sort_values(["area", "year"]).to_csv(out_path, index=False)
                    log(f"  Fetched NASA POWER provinces {len(done_provs) + completed} / {len(df_pag)} total")

    df_clim = pd.DataFrame(all_rows).sort_values(["area", "year"]).reset_index(drop=True)
    df_clim.to_csv(out_path, index=False)
    log(f"  Saved {len(df_clim)} uniform NASA POWER province-year climate rows -> {out_path}")
    return df_clim


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Official PSA OpenSTAT Production & Harvested Area (All Available Crops)
# ─────────────────────────────────────────────────────────────────────────────
def query_psa_table_csv(table_rel_path, crop_targets, period_target="Annual", min_year=1987, max_year=2025):
    url = f"{PSA_BASE}/{table_rel_path}"
    meta = http_get_with_retry(url, timeout=40).json()
    vars_list = meta["variables"]

    crop_var = vars_list[0]
    geo_var = next(v for v in vars_list if v["code"] == "Geolocation")
    yr_var = next(v for v in vars_list if v["code"] == "Year")
    per_var = next(v for v in vars_list if v["code"] == "Period")

    # Select exact crop codes requested
    wanted_low = {k.lower() for k in crop_targets}
    crop_vals = [val for val, txt in zip(crop_var["values"], crop_var["valueTexts"]) if txt.strip().lower() in wanted_low]
    if not crop_vals:
        raise RuntimeError(f"No matching crop targets found in {table_rel_path}: {crop_targets}")

    # Filter out National / Region aggregate headers ("PHILIPPINES", "..REGION ...") to reduce payload & get all provinces
    geo_vals = [
        val for val, txt in zip(geo_var["values"], geo_var["valueTexts"])
        if txt.strip().startswith("....") or "ncr" in txt.lower()
    ]
    yr_vals = [
        val for val, txt in zip(yr_var["values"], yr_var["valueTexts"])
        if txt.strip().isdigit() and min_year <= int(txt.strip()) <= max_year
    ]
    per_vals = [
        val for val, txt in zip(per_var["values"], per_var["valueTexts"])
        if txt.strip().lower() == period_target.lower()
    ]

    # Chunk crops slightly if many crops & many years so PxWeb cell limit (<= 100k cells) is respected
    rows_parsed = []
    chunk_size = 6
    for i in range(0, len(crop_vals), chunk_size):
        c_chunk = crop_vals[i : i + chunk_size]
        query = {
            "query": [
                {"code": crop_var["code"], "selection": {"filter": "item", "values": c_chunk}},
                {"code": geo_var["code"], "selection": {"filter": "item", "values": geo_vals}},
                {"code": yr_var["code"], "selection": {"filter": "item", "values": yr_vals}},
                {"code": per_var["code"], "selection": {"filter": "item", "values": per_vals}},
            ],
            "response": {"format": "csv"},
        }
        resp = http_post_json_with_retry(url, query, timeout=90)
        reader = csv.reader(io.StringIO(resp.text))
        header = next(reader, None)
        if not header:
            continue
        # Header example: ['Crop', 'Geolocation', '2010 Annual', '2011 Annual', ...]
        yr_cols = []
        for idx_col in range(2, len(header)):
            col_yr = int(header[idx_col].split()[0])
            yr_cols.append((idx_col, col_yr))

        for line in reader:
            if len(line) < 3:
                continue
            raw_crop = line[0].strip()
            raw_geo = line[1].strip()
            for idx_col, col_yr in yr_cols:
                val = safe_float(line[idx_col])
                if val is not None:
                    rows_parsed.append((raw_crop, raw_geo, col_yr, val))
        time.sleep(0.25)

    return rows_parsed


def fetch_psa_crop_yields(canon_map):
    out_prod = os.path.join(DATA_DIR, "psa_provincial_production.csv")
    if os.path.exists(out_prod):
        try:
            df_cached = pd.read_csv(out_prod)
            if (
                len(df_cached) >= 30000
                and {"area", "year", "item", "value_hg_ha", "harvested_area_ha"}.issubset(df_cached.columns)
                and "Red Onion" in set(df_cached["item"].dropna())
                and float(df_cached["harvested_area_ha"].min()) >= 1.0
            ):
                log(f"\n[3/5] Loaded cached PSA provincial production ({len(df_cached):,} rows) from {out_prod}")
                return df_cached
        except Exception:
            pass

    log("\n[3/5] Fetching official PSA OpenSTAT provincial crop Production (MT) & Harvested Area (HA)...")

    # Explicit validated pairs: (table_prod, table_area, psa_prod_crop, psa_area_crop, anitaya_canonical_crop, unit_note)
    table_groups = [
        {
            "source_label": "PSA OpenSTAT 0012E4EVCP0.px / 0022E4EAHC0.px",
            "prod_table": "DB/2E/CS/0012E4EVCP0.px",
            "area_table": "DB/2E/CS/0022E4EAHC0.px",
            "min_year": 1987,
            "mappings": [
                ("Palay", "Palay", "Rice, paddy"),
                ("Corn", "Corn", "Maize"),
            ],
        },
        {
            "source_label": "PSA OpenSTAT 0082E4EVCP3.px / 0112E4EAHM3.px",
            "prod_table": "DB/2E/CS/0082E4EVCP3.px",
            "area_table": "DB/2E/CS/0112E4EAHM3.px",
            "min_year": 2010,
            "mappings": [
                ("Cassava, fresh tubers", "Cassava, fresh tubers", "Cassava"),
                ("Camote [Sweet potato]", "Camote [Sweet potato]", "Sweet potatoes"),
                ("Potato", "Potato", "Potatoes"),
                ("Ubi [Greater yam]", "Ubi [Greater yam]", "Yams"),
                ("Gabi [Taro]", "Gabi [Taro]", "Taro"),
                ("Tomato", "Tomato", "Tomato"),
                ("Eggplant", "Eggplant", "Eggplant"),
                ("Ampalaya fruit [Bitter gourd]", "Ampalaya fruit [Bitter gourd]", "Bitter Gourd"),
                ("Squash fruit", "Squash fruit", "Squash"),
                ("Cabbage", "Cabbage", "Cabbage"),
                ("Pechay", "Pechay", "Pechay"),
                ("....Onion, red creole (Bermuda red)", "....Onion, red creole (Bermuda red)", "Red Onion"),
                ("....Onion, yellow granex (Bermuda white)", "....Onion, yellow granex (Bermuda white)", "Yellow Onion"),
                ("Garlic, dried bulb", "Garlic, dried bulb", "Garlic"),
                ("Sitao [Stringbeans]", "Sitao [Stringbeans]", "String Beans"),
                ("Mongo [Mungbean]", "Mongo [Mungbean]", "Mungbean"),
                ("Soybeans", "Soybeans", "Soybeans"),
                ("Peanut, with shell, dry", "Peanut, with shell, dry", "Peanut"),
            ],
        },
        {
            "source_label": "PSA OpenSTAT 0072E4EVCP2.px / 0102E4EAHM2.px",
            "prod_table": "DB/2E/CS/0072E4EVCP2.px",
            "area_table": "DB/2E/CS/0102E4EAHM2.px",
            "min_year": 2010,
            "mappings": [
                ("Banana Saba", "Banana Saba", "Plantains and others"),
                ("Banana Cavendish", "Banana Cavendish", "Banana Cavendish"),
                ("Mango", "Mango", "Mango"),
                ("Pineapple", "Pineapple", "Pineapple"),
                ("Calamansi", "Calamansi", "Calamansi"),
                ("Papaya", "Papaya", "Papaya"),
                ("Watermelon", "Watermelon", "Watermelon"),
            ],
        },
        {
            "source_label": "PSA OpenSTAT 0072E4EVCP1.px / 0092E4EAHM1.px",
            "prod_table": "DB/2E/CS/0072E4EVCP1.px",
            "area_table": "DB/2E/CS/0092E4EAHM1.px",
            "min_year": 2010,
            "mappings": [
                ("Sorghum", "Sorghum", "Sorghum"),
                ("Coconut (w/ husk)", "Coconut", "Coconut"),
                ("Sugarcane", "Sugarcane", "Sugarcane"),
                ("Coffee (green coffee beans)", "Coffee", "Coffee"),
                ("Cacao", "Cacao", "Cacao"),
                ("Abaca (dried raw fiber)", "Abaca", "Abaca"),
                ("Tobacco", "Tobacco", "Tobacco"),
                ("Rubber (coagulated cup lump)", "Rubber", "Rubber"),
            ],
        },
    ]

    aggregated_records = {}  # key: (province, year, canonical_crop) -> {prod_mt, area_ha, source}

    for grp in table_groups:
        p_targets = [m[0] for m in grp["mappings"]]
        a_targets = [m[1] for m in grp["mappings"]]
        prod_to_canon = {m[0].lower(): m[2] for m in grp["mappings"]}
        area_to_canon = {m[1].lower(): m[2] for m in grp["mappings"]}

        log(f"  Querying {grp['source_label']} ({len(grp['mappings'])} crops)...")
        prod_rows = query_psa_table_csv(grp["prod_table"], p_targets, min_year=grp["min_year"], max_year=2025)
        area_rows = query_psa_table_csv(grp["area_table"], a_targets, min_year=grp["min_year"], max_year=2025)

        # Accumulate into dict; handle Maguindanao split sums & avoid Negros a/ duplicate counts
        prod_map = {}
        for rc, rg, yr, val in prod_rows:
            prov = resolve_psa_province(rg, canon_map)
            canon_c = prod_to_canon.get(rc.lower())
            if not prov or not canon_c or val <= 0:
                continue
            raw_geo_low = norm_text(rg).lower()
            # Keep exact split tracking for Maguindanao del Norte + del Sur
            k = (prov, yr, canon_c, raw_geo_low if "maguindanao" in raw_geo_low else prov.lower())
            prod_map[k] = val

        # Collapse sub-province keys (sum Maguindanao halves if both exist for the same year)
        prod_collapsed = {}
        for (prov, yr, canon_c, _sub), val in prod_map.items():
            k2 = (prov, yr, canon_c)
            prod_collapsed[k2] = prod_collapsed.get(k2, 0.0) + val

        area_map = {}
        for rc, rg, yr, val in area_rows:
            prov = resolve_psa_province(rg, canon_map)
            canon_c = area_to_canon.get(rc.lower())
            if not prov or not canon_c or val <= 0:
                continue
            raw_geo_low = norm_text(rg).lower()
            k = (prov, yr, canon_c, raw_geo_low if "maguindanao" in raw_geo_low else prov.lower())
            area_map[k] = val

        area_collapsed = {}
        for (prov, yr, canon_c, _sub), val in area_map.items():
            k2 = (prov, yr, canon_c)
            area_collapsed[k2] = area_collapsed.get(k2, 0.0) + val

        matched_cnt = 0
        for (prov, yr, canon_c), p_mt in prod_collapsed.items():
            a_ha = area_collapsed.get((prov, yr, canon_c))
            # Filter sub-hectare backyard area (< 1.0 ha) to prevent PSA small-denominator division spikes
            if a_ha is None or a_ha < 1.0 or p_mt <= 0:
                continue
            # Strictly compute yield from compatible MT / HA units
            yield_mt_ha = p_mt / a_ha
            value_hg_ha = yield_mt_ha * 10000.0
            # Filter physically impossible transcription errors (e.g. > 200 MT/ha or < 0.02 MT/ha)
            if yield_mt_ha < 0.02 or yield_mt_ha > 200.0:
                continue
            aggregated_records[(prov, yr, canon_c)] = {
                "area": prov,
                "region": canon_map[prov.lower()]["region"],
                "year": int(yr),
                "item": canon_c,
                "production_mt": round(p_mt, 2),
                "harvested_area_ha": round(a_ha, 2),
                "yield_mt_ha": round(yield_mt_ha, 4),
                "value_hg_ha": round(value_hg_ha, 1),
                "production_source": grp["source_label"],
            }
            matched_cnt += 1
        log(f"    -> Valid matched (province, year, crop) observations: {matched_cnt:,}")

    df_prod = pd.DataFrame(list(aggregated_records.values()))
    df_prod = df_prod.sort_values(["item", "area", "year"]).reset_index(drop=True)
    out_prod = os.path.join(DATA_DIR, "psa_provincial_production.csv")
    df_prod.to_csv(out_prod, index=False)
    log(f"  Saved {len(df_prod):,} official PSA provincial production rows -> {out_prod}")
    return df_prod


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Official PSA Farmgate Prices & Palay Costs & Returns
# ─────────────────────────────────────────────────────────────────────────────
def fetch_psa_farmgate_prices(canon_map):
    log("\n[4/5] Fetching official PSA OpenSTAT provincial annual farmgate prices (2010-2025)...")
    nfg_specs = [
        ("DB/2M/NFG/0032M4AFN01.px", {
            "Palay [Paddy] Other Variety, dry (conv. to 14% mc)": ("Rice, paddy", "PHP/kg"),
            "Corngrain [Maize] Yellow, matured": ("Maize", "PHP/kg"),
        }),
        ("DB/2M/NFG/0032M4AFN02.px", {
            "Cassava for food, fresh tubers": ("Cassava", "PHP/kg"),
            "Camote [Sweet potato]": ("Sweet potatoes", "PHP/kg"),
            "Potato": ("Potatoes", "PHP/kg"),
            "Ubi [Greater yam]": ("Yams", "PHP/kg"),
            "Gabi for sinigang, Tagalog [Taro]": ("Taro", "PHP/kg"),
        }),
        ("DB/2M/NFG/0032M4AFN03.px", {
            "Sitao [Stringbeans]": ("String Beans", "PHP/kg"),
            "Mongo [Mungbean], green, shiny": ("Mungbean", "PHP/kg"),
            "Soybeans": ("Soybeans", "PHP/kg"),
            "Peanut, with shell, dry": ("Peanut", "PHP/kg"),
        }),
        ("DB/2M/NFG/0032M4AFN04.px", {
            "Garlic, dried bulb": ("Garlic", "PHP/kg"),
            "Onion, red creole (Bermuda red)": ("Red Onion", "PHP/kg"),
            "Onion, yellow granex (Bermuda white)": ("Yellow Onion", "PHP/kg"),
        }),
        ("DB/2M/NFG/0032M4AFN05.px", {
            "Ampalaya fruit [Bitter gourd]": ("Bitter Gourd", "PHP/kg"),
            "Eggplant, long, purple": ("Eggplant", "PHP/kg"),
            "Squash fruit": ("Squash", "PHP/kg"),
            "Tomato": ("Tomato", "PHP/kg"),
        }),
        ("DB/2M/NFG/0032M4AFN06.px", {
            "Cabbage": ("Cabbage", "PHP/kg"),
            "Pechay, native": ("Pechay", "PHP/kg"),
        }),
        ("DB/2M/NFG/0032M4AFN07.px", {
            "Banana Saba, green": ("Plantains and others", "PHP/kg"),
            "Banana Cavendish": ("Banana Cavendish", "PHP/kg"),
            "Calamansi": ("Calamansi", "PHP/kg"),
            "Mango Carabao, green": ("Mango", "PHP/kg"),
            "Papaya Solo": ("Papaya", "PHP/kg"),
            "Pineapple Hawaiian": ("Pineapple", "PHP/kg"),
            "Watermelon": ("Watermelon", "PHP/kg"),
        }),
        ("DB/2M/NFG/0032M4AFN08.px", {
            "Abaca": ("Abaca", "PHP/kg"),
            "Cacao, dry beans [Cocoa beans]": ("Cacao", "PHP/kg"),
            "Coconut matured": ("Coconut", "PHP/nut (PSA Official)"),
            "Coffee Robusta, green coffee beans": ("Coffee", "PHP/kg"),
            "Rubber Cuplump": ("Rubber", "PHP/kg"),
            "Sorghum": ("Sorghum", "PHP/kg"),
            "Sugarcane for Chewing": ("Sugarcane", "PHP/kg (cane)"),
            "Tobacco Native, dry": ("Tobacco", "PHP/kg"),
        }),
    ]

    price_records = {}  # (prov, yr, crop) -> record
    for tid, comm_map in nfg_specs:
        url = f"{PSA_BASE}/{tid}"
        meta = http_get_with_retry(url, timeout=40).json()
        vars_list = meta["variables"]
        geo_var = next(v for v in vars_list if v["code"] == "Geolocation")
        comm_var = next(v for v in vars_list if v["code"] == "Commodity")
        yr_var = next(v for v in vars_list if v["code"] == "Year")
        per_var = next(v for v in vars_list if v["code"] == "Period")

        wanted_low = {k.lower(): (v[0], v[1]) for k, v in comm_map.items()}
        comm_vals = [val for val, txt in zip(comm_var["values"], comm_var["valueTexts"]) if txt.strip().lower() in wanted_low]
        geo_vals = [val for val, txt in zip(geo_var["values"], geo_var["valueTexts"]) if txt.strip().startswith("....") or "ncr" in txt.lower()]
        yr_vals = [val for val, txt in zip(yr_var["values"], yr_var["valueTexts"]) if txt.strip().isdigit() and 2010 <= int(txt.strip()) <= 2025]
        per_vals = [val for val, txt in zip(per_var["values"], per_var["valueTexts"]) if txt.strip().lower() == "annual"]

        if not (comm_vals and yr_vals and per_vals):
            continue

        query = {
            "query": [
                {"code": geo_var["code"], "selection": {"filter": "item", "values": geo_vals}},
                {"code": comm_var["code"], "selection": {"filter": "item", "values": comm_vals}},
                {"code": yr_var["code"], "selection": {"filter": "item", "values": yr_vals}},
                {"code": per_var["code"], "selection": {"filter": "item", "values": per_vals}},
            ],
            "response": {"format": "csv"},
        }
        resp_csv = http_post_json_with_retry(url, query, timeout=60).text
        reader = csv.reader(io.StringIO(resp_csv))
        header = next(reader, None)
        if not header:
            continue

        # Two PxWeb CSV layouts exist in NFG tables:
        # Layout A (AFN01..07): ['Geolocation', 'Commodity', '2010 Annual', ...]
        # Layout B (AFN08):     ['Geolocation', '2010 Annual Abaca', '2010 Annual Cacao, ...', ...]
        has_comm_col = len(header) >= 2 and header[1].strip().lower() == "commodity"
        cnt_table = 0

        if has_comm_col:
            yr_cols = []
            for idx_col in range(2, len(header)):
                yr_token = header[idx_col].strip().split()[0]
                if yr_token.isdigit():
                    yr_cols.append((idx_col, int(yr_token)))

            for line in reader:
                if len(line) < 3:
                    continue
                prov = resolve_psa_province(line[0].strip(), canon_map)
                if not prov:
                    continue
                mapped = wanted_low.get(line[1].strip().lower())
                if not mapped:
                    continue
                canon_crop, unit_str = mapped
                for idx_col, yr in yr_cols:
                    val = safe_float(line[idx_col])
                    if val is None or val <= 0:
                        continue
                    price_records[(prov, yr, canon_crop)] = {
                        "Province": prov,
                        "Region": canon_map[prov.lower()]["region"],
                        "Crop_Name": canon_crop,
                        "Year": yr,
                        "Farmgate_Price": round(val, 2),
                        "Price_Unit": unit_str,
                        "Price_Source": f"PSA OpenSTAT {tid}",
                    }
                    cnt_table += 1
        else:
            parsed_cols = []
            for idx_col in range(1, len(header)):
                col_txt = header[idx_col].strip()
                if " annual " not in col_txt.lower():
                    continue
                yr_part, comm_part = col_txt.split(" Annual ", 1) if " Annual " in col_txt else (col_txt[:4], col_txt[12:])
                if not yr_part.strip().isdigit():
                    continue
                mapped = wanted_low.get(comm_part.strip().lower())
                if mapped:
                    parsed_cols.append((idx_col, int(yr_part.strip()), mapped[0], mapped[1]))

            for line in reader:
                if len(line) < 2:
                    continue
                prov = resolve_psa_province(line[0].strip(), canon_map)
                if not prov:
                    continue
                for idx_col, yr, canon_crop, unit_str in parsed_cols:
                    if idx_col >= len(line):
                        continue
                    val = safe_float(line[idx_col])
                    if val is None or val <= 0:
                        continue
                    price_records[(prov, yr, canon_crop)] = {
                        "Province": prov,
                        "Region": canon_map[prov.lower()]["region"],
                        "Crop_Name": canon_crop,
                        "Year": yr,
                        "Farmgate_Price": round(val, 2),
                        "Price_Unit": unit_str,
                        "Price_Source": f"PSA OpenSTAT {tid}",
                    }
                    cnt_table += 1
        log(f"  {tid}: collected {cnt_table:,} valid provincial annual prices")
        time.sleep(0.25)

    rows_fg = list(price_records.values())
    df_fg = (
        pd.DataFrame(rows_fg).sort_values(["Crop_Name", "Province", "Year"]).reset_index(drop=True)
        if rows_fg
        else pd.DataFrame(columns=["Province", "Region", "Crop_Name", "Year", "Farmgate_Price", "Price_Unit", "Price_Source"])
    )
    out_fg = os.path.join(DATA_DIR, "psa_provincial_farmgate_prices.csv")
    df_fg.to_csv(out_fg, index=False)
    log(f"  Saved {len(df_fg):,} official PSA farmgate prices -> {out_fg}")

    # Also fetch Table 5.5/5.6/5.7 Palay Costs & Returns (2022-2024 PhP/ha) using CSV format
    cost_rows = []
    for tid, eco in [
        ("DB/3S/C5/0052E2BARP0.px", "All Palay"),
        ("DB/3S/C5/0062E2BARI0.px", "Irrigated Palay"),
        ("DB/3S/C5/0072E2BARN0.px", "Non-Irrigated Palay"),
    ]:
        url = f"{PSA_BASE}/{tid}"
        meta = http_get_with_retry(url, timeout=30).json()
        q = {"query": [{"code": v["code"], "selection": {"filter": "item", "values": v["values"]}} for v in meta["variables"]], "response": {"format": "csv"}}
        resp_csv = http_post_json_with_retry(url, q, timeout=30).text
        reader = csv.reader(io.StringIO(resp_csv))
        header = next(reader, None)
        if not header:
            continue
        # Header: ['Item', '2022 January to June', '2022 July to December', '2022 Average', ...]
        col_meta = []
        for idx_col in range(1, len(header)):
            parts = header[idx_col].strip().split(maxsplit=1)
            if parts and parts[0].isdigit():
                col_meta.append((idx_col, int(parts[0]), parts[1] if len(parts) > 1 else "Average"))
        for line in reader:
            if len(line) < 2:
                continue
            item_lbl = line[0].strip()
            for idx_col, yr_val, sem_lbl in col_meta:
                fv = safe_float(line[idx_col])
                if fv is not None:
                    cost_rows.append({
                        "Ecosystem": eco,
                        "Item": item_lbl,
                        "Year": yr_val,
                        "Semester": sem_lbl,
                        "Value": fv,
                        "Source": f"PSA OpenSTAT {tid}",
                    })
    df_palay_cr = pd.DataFrame(cost_rows)
    out_cr = os.path.join(DATA_DIR, "psa_palay_costs_returns.csv")
    df_palay_cr.to_csv(out_cr, index=False)
    log(f"  Saved {len(df_palay_cr)} PSA Table 5.5/5.6/5.7 Palay Costs & Returns rows -> {out_cr}")
    return df_fg, df_palay_cr


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Merge Real PH Dataset, Sync Provincial Normals & Optional Manual Ingest
# ─────────────────────────────────────────────────────────────────────────────
def ingest_manual_imports_if_present(canon_map):
    """
    Hook for any manual CSV exports placed by the researcher into data/manual_imports/:
      - PhilRice PalayStat provincial rice yields
      - PAGASA station-level annual rainfall/temperature records
    Must carry columns: area (or province), year, item, value_hg_ha (or yield_mt_ha).
    """
    extra_rows = []
    if not os.path.exists(MANUAL_DIR):
        return extra_rows
    for fname in sorted(os.listdir(MANUAL_DIR)):
        if not fname.lower().endswith(".csv"):
            continue
        fpath = os.path.join(MANUAL_DIR, fname)
        try:
            dm = pd.read_csv(fpath)
            cols_low = {c.lower().strip(): c for c in dm.columns}
            p_col = cols_low.get("area") or cols_low.get("province")
            y_col = cols_low.get("year")
            i_col = cols_low.get("item") or cols_low.get("crop_name")
            if not (p_col and y_col and i_col):
                continue
            for _, r in dm.iterrows():
                prov = resolve_psa_province(r[p_col], canon_map)
                yr = int(r[y_col]) if pd.notna(r[y_col]) else None
                if not prov or not yr:
                    continue
                vhg = safe_float(r.get(cols_low.get("value_hg_ha")))
                ymt = safe_float(r.get(cols_low.get("yield_mt_ha")))
                pmt = safe_float(r.get(cols_low.get("production_mt")))
                aha = safe_float(r.get(cols_low.get("harvested_area_ha")))
                if vhg is None and ymt is not None and ymt > 0:
                    vhg = ymt * 10000.0
                elif vhg is None and pmt and aha and pmt > 0 and aha > 0:
                    vhg = (pmt / aha) * 10000.0
                if vhg and vhg > 0:
                    extra_rows.append({
                        "area": prov,
                        "region": canon_map[prov.lower()]["region"],
                        "year": yr,
                        "item": str(r[i_col]).strip(),
                        "production_mt": round(pmt, 2) if pmt else np.nan,
                        "harvested_area_ha": round(aha, 2) if aha else np.nan,
                        "yield_mt_ha": round(vhg / 10000.0, 4),
                        "value_hg_ha": round(vhg, 1),
                        "production_source": f"Manual Import ({fname})",
                    })
        except Exception as e:
            log(f"  Warning reading manual import {fname}: {e}")
    return extra_rows


def build_final_dataset():
    df_suit, canon_map = load_canonical_provinces()

    # Run Step 1-4
    df_pag = fetch_pagasa_climap_normals(canon_map)
    df_clim = fetch_yearly_historical_climate(df_pag, start_year=1987, end_year=2025)
    df_prod = fetch_psa_crop_yields(canon_map)
    df_fg, _ = fetch_psa_farmgate_prices(canon_map)

    manual_rows = ingest_manual_imports_if_present(canon_map)
    if manual_rows:
        df_prod = pd.concat([df_prod, pd.DataFrame(manual_rows)], ignore_index=True)

    log("\n[5/5] Merging Philippine Yield + Yearly Climate + PAGASA Normals into data/philippines_crop_yield_climate.csv...")
    df_merged = df_prod.merge(df_clim, on=["area", "year"], how="inner")
    df_merged = df_merged.merge(
        df_pag[["province", "pagasa_normal_rain_mm", "pagasa_normal_temp_c", "centroid_lat", "centroid_lon"]].rename(columns={"province": "area"}),
        on="area",
        how="left",
    )

    # Deduplicate cleanly so zero identical (area, year, item) records exist across train/test
    before_dedup = len(df_merged)
    df_merged = df_merged.dropna(subset=["area", "year", "item", "average_rain_fall_mm_per_year", "avg_temp", "value_hg_ha"])
    df_merged = df_merged.drop_duplicates(subset=["area", "year", "item"], keep="first").sort_values(["item", "area", "year"]).reset_index(drop=True)
    log(f"  Merged records: {before_dedup:,} -> Clean deduplicated records: {len(df_merged):,}")

    # Cross-check correlation between NASA POWER (1991-2020) and PAGASA CliMap v2.1 (1991-2020)
    era_91_20 = df_clim[(df_clim["year"] >= 1991) & (df_clim["year"] <= 2020)].groupby("area", as_index=False)[["average_rain_fall_mm_per_year", "avg_temp"]].mean()
    comp_clim = era_91_20.merge(df_pag[["province", "pagasa_normal_rain_mm", "pagasa_normal_temp_c"]], left_on="area", right_on="province")
    r_corr_rain = comp_clim["average_rain_fall_mm_per_year"].corr(comp_clim["pagasa_normal_rain_mm"])
    r_corr_temp = comp_clim["avg_temp"].corr(comp_clim["pagasa_normal_temp_c"])
    log(f"  Climate Validation (NASA POWER MERRA-2 1991-2020 vs DOST-PAGASA CliMap v2.1 GeoTIFF): rain corr r={r_corr_rain:.3f}, temp corr r={r_corr_temp:.3f}")

    out_ml = os.path.join(DATA_DIR, "philippines_crop_yield_climate.csv")
    df_merged.to_csv(out_ml, index=False)
    log(f"  SAVED MASTER PHILIPPINE DATASET: {out_ml}")

    # Update philippine_provinces_crop_suitability.csv & .json with official PAGASA CliMap normals & coords
    pag_lookup = df_pag.set_index("province").to_dict(orient="index")
    updated_rain, updated_temp, updated_lat, updated_lon, updated_wet, updated_dry, clim_sources = [], [], [], [], [], [], []
    for _, row in df_suit.iterrows():
        p_info = pag_lookup.get(str(row["province"]).strip())
        if p_info and pd.notna(p_info.get("pagasa_normal_rain_mm")) and pd.notna(p_info.get("pagasa_normal_temp_c")):
            updated_rain.append(round(float(p_info["pagasa_normal_rain_mm"]), 1))
            updated_temp.append(round(float(p_info["pagasa_normal_temp_c"]), 2))
            updated_lat.append(p_info["centroid_lat"])
            updated_lon.append(p_info["centroid_lon"])
            updated_wet.append(p_info["pagasa_wet_rain_mm"])
            updated_dry.append(p_info["pagasa_dry_rain_mm"])
            clim_sources.append("DOST-PAGASA CliMap v2.1 Observed Normals (1991-2020)")
        else:
            updated_rain.append(row["annual_rainfall_mm"])
            updated_temp.append(row["avg_temp_c"])
            updated_lat.append(np.nan)
            updated_lon.append(np.nan)
            updated_wet.append(np.nan)
            updated_dry.append(np.nan)
            clim_sources.append("Legacy Reference Estimate")

    df_suit["annual_rainfall_mm"] = updated_rain
    df_suit["avg_temp_c"] = updated_temp
    df_suit["centroid_lat"] = updated_lat
    df_suit["centroid_lon"] = updated_lon
    df_suit["pagasa_wet_rain_mm"] = updated_wet
    df_suit["pagasa_dry_rain_mm"] = updated_dry
    df_suit["climate_data_source"] = clim_sources

    suit_csv_path = os.path.join(DATA_DIR, "philippine_provinces_crop_suitability.csv")
    suit_json_path = os.path.join(DATA_DIR, "philippine_provinces_crop_suitability.json")
    df_suit.to_csv(suit_csv_path, index=False)
    with open(suit_json_path, "w", encoding="utf-8") as f:
        json.dump(df_suit.to_dict(orient="records"), f, indent=2)
    log(f"  Synchronized 82-province suitability CSV & JSON with official PAGASA CliMap normals -> {suit_csv_path}")

    # Summary statistics
    print("\n" + "=" * 75)
    print("PHILIPPINE DATASET SUMMARY (philippines_crop_yield_climate.csv)")
    print("=" * 75)
    print(f"Total clean observations: {len(df_merged):,}")
    print(f"Provinces covered:        {df_merged['area'].nunique()} provinces")
    print(f"Crops covered:            {df_merged['item'].nunique()} crops")
    print(f"Year range:               {df_merged['year'].min()} - {df_merged['year'].max()}")
    print("\nObservations per crop:")
    print(df_merged["item"].value_counts().to_string())


if __name__ == "__main__":
    build_final_dataset()
