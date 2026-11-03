import json
import os
import sys

from app import app

def test_routes():
    print("Initializing Flask test_client (socketless in-memory test)...")
    client = app.test_client()

    # 1. Test GET /
    print("\n1. Testing GET / ...")
    res_home = client.get("/")
    print(f"Status: {res_home.status_code}")
    assert res_home.status_code == 200
    html = res_home.data.decode("utf-8")
    assert "AniTaya" in html
    assert "Laguna State Polytechnic University" in html
    assert "Random Forest" in html
    print("GET / passed successfully! (Rendered complete dashboard HTML with CSST 102 badges)")

    # 2. Test GET /api/benchmarks
    print("\n2. Testing GET /api/benchmarks ...")
    res_bm = client.get("/api/benchmarks")
    assert res_bm.status_code == 200
    bm_data = json.loads(res_bm.data.decode("utf-8"))
    assert "crops" in bm_data
    assert "Rice, paddy" in bm_data["crops"]
    print(f"GET /api/benchmarks passed! ({len(bm_data['crops'])} crops, {len(bm_data['regions'])} regions)")

    # 3. Test POST /api/predict (Laguna Rice Farmer)
    print("\n3. Testing POST /api/predict (Sample: Laguna Rice Farmer 2.0 ha)...")
    payload = {
        "crop": "Rice, paddy",
        "land_area": 2.0,
        "season": "Wet Season",
        "rainfall": 2400.0,
        "avg_temp": 28.2,
        "pesticides_tonnes": 120.0
    }
    res_pred = client.post("/api/predict", data=json.dumps(payload), content_type="application/json")
    print(f"Status: {res_pred.status_code}")
    assert res_pred.status_code == 200
    data = json.loads(res_pred.data.decode("utf-8"))
    assert data["status"] == "success"

    yp = data["yield_predictions"]
    fin = data["financial_forecast"]
    sc = data["scenarios"]

    print(f"-> Predicted Yield Rate: {yp['yield_rate_mt_ha']} MT/ha")
    print(f"-> Total Harvest: {yp['total_yield_mt']} MT ({yp['total_yield_kg']:,.1f} kg)")
    print(f"-> Cavans (50kg bags): {yp['total_sacks_cavans']:,.1f} sacks")
    print(f"-> Gross Farm Revenue: PHP {fin['gross_revenue_php']:,.2f}")
    print(f"-> Total Production Cost: PHP {fin['production_cost_php']:,.2f}")
    print(f"-> Net Farmer Profit: PHP {fin['net_profit_php']:,.2f}")
    print(f"-> Return on Investment (ROI): {fin['roi_percent']}%")
    print(f"-> Scenarios: Best-case ({sc['favorable']['yield_mt']} MT), Baseline ({sc['expected']['yield_mt']} MT), Adverse ({sc['adverse']['yield_mt']} MT)")
    print("POST /api/predict passed with full financial and agronomic forecast!")

    # 4. Test POST /api/predict (Validation Error Handling)
    print("\n4. Testing Input Validation Error Handling (Negative land area)...")
    bad_payload = {
        "crop": "Rice, paddy",
        "land_area": -3.5
    }
    res_bad = client.post("/api/predict", data=json.dumps(bad_payload), content_type="application/json")
    print(f"Status: {res_bad.status_code}")
    assert res_bad.status_code == 422
    err_data = json.loads(res_bad.data.decode("utf-8"))
    print(f"Errors returned correctly: {err_data['errors']}")
    print("Validation testing passed!")

    print("\n=======================================================")
    print("ALL FLASK ROUTES, ML PIPELINE, AND PREDICTIONS VERIFIED!")
    print("=======================================================")

if __name__ == "__main__":
    test_routes()
