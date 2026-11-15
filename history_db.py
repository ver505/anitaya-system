"""
Taya System - Prediction History Database Engine (SQLite)
Provides persistent storage, retrieval, and deletion of multi-crop prediction records.
"""

import os
import sqlite3
import json
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "taya_history.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS prediction_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        province TEXT NOT NULL,
        region TEXT NOT NULL,
        season TEXT NOT NULL,
        soil_type TEXT,
        crop_count INTEGER NOT NULL,
        total_land_area REAL NOT NULL,
        total_gross_income REAL NOT NULL,
        total_production_cost REAL NOT NULL,
        total_net_income REAL NOT NULL,
        best_crop_name TEXT NOT NULL,
        crops_data_json TEXT NOT NULL
    )
    """)
    _backfill_unspecified_provinces(conn, cursor)
    conn.commit()
    conn.close()


def _extract_prediction_metadata(data):
    """Extract normalized location and season metadata from flat or nested calculate payloads."""
    if not isinstance(data, dict):
        return {
            "province": "Unspecified Province",
            "region": "",
            "season": "Wet Season",
            "soil_type": "Loam",
        }

    params = data.get("parameters") if isinstance(data.get("parameters"), dict) else {}
    prov_details = data.get("province_details") if isinstance(data.get("province_details"), dict) else {}

    top_province = str(data.get("province") or "").strip()
    if top_province.lower() == "unspecified province":
        top_province = ""

    province = (
        top_province
        or str(params.get("province") or "").strip()
        or str(prov_details.get("province") or "").strip()
        or "Unspecified Province"
    )
    region = (
        str(data.get("region") or "").strip()
        or str(params.get("region") or "").strip()
        or str(prov_details.get("region") or "").strip()
    )
    season = (
        str(data.get("season") or "").strip()
        or str(params.get("season") or "").strip()
        or "Wet Season"
    )
    soil_type = (
        str(data.get("soil_type") or "").strip()
        or str(params.get("soil_type") or "").strip()
        or str(prov_details.get("dominant_soil") or "").strip()
        or "Loam"
    )
    return {
        "province": province,
        "region": region,
        "season": season,
        "soil_type": soil_type,
    }


def _backfill_unspecified_provinces(conn, cursor):
    """Repair legacy rows that stored 'Unspecified Province' when nested parameters hold the province."""
    rows = cursor.execute("""
        SELECT id, province, region, season, soil_type, crops_data_json
        FROM prediction_history
        WHERE province IS NULL OR TRIM(province) = '' OR province = 'Unspecified Province'
    """).fetchall()
    for row in rows:
        try:
            payload = json.loads(row["crops_data_json"])
        except Exception:
            continue
        meta = _extract_prediction_metadata(payload)
        if meta["province"] != "Unspecified Province":
            cursor.execute("""
                UPDATE prediction_history
                SET province = ?, region = ?, season = ?, soil_type = ?
                WHERE id = ?
            """, (meta["province"], meta["region"], meta["season"], meta["soil_type"], row["id"]))


def save_prediction(data):
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta = _extract_prediction_metadata(data)
    province = meta["province"]
    region = meta["region"]
    season = meta["season"]
    soil_type = meta["soil_type"]
    crop_results = data.get("crop_results", [])
    
    crop_count = len(crop_results)
    total_area = sum(float(c.get("land_area_ha", 1.0)) for c in crop_results) if crop_results else 0.0
    total_gross = sum(float(c.get("gross_income_php", 0.0)) for c in crop_results) if crop_results else 0.0
    total_cost = sum(float(c.get("production_cost_php", 0.0)) for c in crop_results) if crop_results else 0.0
    total_net = sum(float(c.get("net_income_php", 0.0)) for c in crop_results) if crop_results else 0.0
    
    # Best crop by net income
    best_crop = ""
    if crop_results:
        best_item = max(crop_results, key=lambda x: float(x.get("net_income_php", -1e9)))
        best_crop = best_item.get("crop_name", "")
        
    crops_json = json.dumps(data)
    
    cursor.execute("""
    INSERT INTO prediction_history (
        created_at, province, region, season, soil_type,
        crop_count, total_land_area, total_gross_income, total_production_cost,
        total_net_income, best_crop_name, crops_data_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        now_iso, province, region, season, soil_type,
        crop_count, total_area, total_gross, total_cost,
        total_net, best_crop, crops_json
    ))
    
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return record_id

def get_history(limit=50):
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT id, created_at, province, region, season, soil_type,
           crop_count, total_land_area, total_gross_income, total_production_cost,
           total_net_income, best_crop_name
    FROM prediction_history
    ORDER BY id DESC
    LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    
    result = []
    for row in rows:
        result.append({
            "id": row["id"],
            "created_at": row["created_at"],
            "province": row["province"],
            "region": row["region"],
            "season": row["season"],
            "soil_type": row["soil_type"],
            "crop_count": row["crop_count"],
            "total_land_area": row["total_land_area"],
            "total_gross_income": row["total_gross_income"],
            "total_production_cost": row["total_production_cost"],
            "total_net_income": row["total_net_income"],
            "best_crop_name": row["best_crop_name"]
        })
    return result

def get_prediction_detail(record_id):
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM prediction_history WHERE id = ?", (record_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return None
        
    data = json.loads(row["crops_data_json"])
    data["id"] = row["id"]
    data["created_at"] = row["created_at"]
    return data

def delete_prediction(record_id):
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM prediction_history WHERE id = ?", (record_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted

def clear_all_history():
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM prediction_history")
    conn.commit()
    conn.close()
    return True

def get_stats():
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) AS total_count FROM prediction_history")
    row = cursor.fetchone()
    conn.close()
    return {"total_saved_predictions": row["total_count"] if row else 0}

if __name__ == "__main__":
    init_db()
    print("Database initialized at:", DB_PATH)
    print("Current stats:", get_stats())
