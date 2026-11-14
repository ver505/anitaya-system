import json
from app import app

def test_new_routes():
    client = app.test_client()

    print("Testing GET /api/crops ...")
    res = client.get("/api/crops")
    assert res.status_code == 200
    data = json.loads(res.data.decode("utf-8"))
    assert data["status"] == "success"
    assert data["count"] >= 35
    print(f"GET /api/crops passed! ({data['count']} crops)")

    print("\nTesting GET /api/crops/suitability/Nueva Ecija ...")
    res = client.get("/api/crops/suitability/Nueva Ecija")
    assert res.status_code == 200
    data = json.loads(res.data.decode("utf-8"))
    assert data["province"] == "Nueva Ecija"
    rec_count = sum(1 for c in data["data"] if c.get("is_recommended"))
    print(f"Suitability endpoint passed! ({rec_count} recommended crops in Nueva Ecija)")

    print("\nTesting POST /api/calculate (Multi-crop: Rice + Onion) ...")
    payload = {
        "province": "Nueva Ecija",
        "season": "Dry Season",
        "soil_type": "Clay Loam",
        "crops": [
            {"crop_name": "Rice, paddy", "land_area_ha": 2.5},
            {"crop_name": "Red Onion", "land_area_ha": 1.0}
        ]
    }
    res = client.post("/api/calculate", data=json.dumps(payload), content_type="application/json")
    assert res.status_code == 200
    data = json.loads(res.data.decode("utf-8"))
    assert data["status"] == "success"
    assert len(data["crop_results"]) == 2
    assert "comparison_summary" in data
    print(f"POST /api/calculate passed! Total Net Income: PHP {data['totals']['total_net_income_php']:,.2f}")

    print("\nTesting POST /api/history/save ...")
    res_save = client.post("/api/history/save", data=json.dumps(data), content_type="application/json")
    assert res_save.status_code == 200
    saved = json.loads(res_save.data.decode("utf-8"))
    record_id = saved["id"]
    print(f"POST /api/history/save passed! Record ID: {record_id}")

    print("\nTesting GET /api/history ...")
    res_hist = client.get("/api/history")
    assert res_hist.status_code == 200
    hist = json.loads(res_hist.data.decode("utf-8"))
    assert len(hist["history"]) >= 1
    print(f"GET /api/history passed! ({len(hist['history'])} records)")

    print(f"\nTesting GET /api/history/{record_id} ...")
    res_rec = client.get(f"/api/history/{record_id}")
    assert res_rec.status_code == 200
    rec_detail = json.loads(res_rec.data.decode("utf-8"))
    assert rec_detail["status"] == "success"
    print(f"GET /api/history/{record_id} passed!")

    print(f"\nTesting DELETE /api/history/{record_id} ...")
    res_del = client.delete(f"/api/history/{record_id}")
    assert res_del.status_code == 200
    print(f"DELETE /api/history/{record_id} passed!")

    print("\nTesting GET /api/stats ...")
    res_stats = client.get("/api/stats")
    assert res_stats.status_code == 200
    stats = json.loads(res_stats.data.decode("utf-8"))["stats"]
    assert stats["total_crops"] >= 35
    print(f"GET /api/stats passed! Stats: {stats}")

    print("\nALL NEW API ROUTES VERIFIED SUCCESSFULLY!")

if __name__ == "__main__":
    test_new_routes()
