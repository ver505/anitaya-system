import time
import subprocess
import requests
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def test():
    print("Starting Flask app in background...")
    proc = subprocess.Popen([sys.executable, "app.py"], cwd=BASE_DIR)
    time.sleep(3)

    try:
        # Test Home Route
        print("Testing GET / ...")
        r_home = requests.get("http://127.0.0.1:5000/", timeout=5)
        print(f"Home status code: {r_home.status_code}")
        assert r_home.status_code == 200
        assert "AniTaya" in r_home.text
        print("GET / passed!")

        # Test Predict Route
        print("\nTesting POST /api/predict ...")
        payload = {
            "crop": "Rice, paddy",
            "land_area": 2.0,
            "season": "Wet Season",
            "rainfall": 2400.0,
            "avg_temp": 28.2,
            "pesticides_tonnes": 120.0
        }
        r_pred = requests.post("http://127.0.0.1:5000/api/predict", json=payload, timeout=5)
        print(f"API status code: {r_pred.status_code}")
        assert r_pred.status_code == 200
        res = r_pred.json()
        print(f"Status: {res['status']}")
        print(f"Crop: {res['inputs']['crop_filipino']}")
        print(f"Total Yield (MT): {res['yield_predictions']['total_yield_mt']} MT")
        print(f"Total Cavans (50kg bags): {res['yield_predictions']['total_sacks_cavans']} sacks")
        print(f"Gross Revenue: PHP {res['financial_forecast']['gross_revenue_php']:,.2f}")
        print(f"Total Production Cost: PHP {res['financial_forecast']['production_cost_php']:,.2f}")
        print(f"Net Profit: PHP {res['financial_forecast']['net_profit_php']:,.2f}")
        print(f"ROI: {res['financial_forecast']['roi_percent']}%")
        print("POST /api/predict passed!")

        # Test Invalid Input Handling
        print("\nTesting Input Validation Error Handling...")
        bad_payload = {"crop": "Rice, paddy", "land_area": -5.0}
        r_bad = requests.post("http://127.0.0.1:5000/api/predict", json=bad_payload, timeout=5)
        print(f"Validation status code: {r_bad.status_code}")
        assert r_bad.status_code == 422
        print(f"Handled error response: {r_bad.json()['errors']}")
        print("Validation test passed!")

    finally:
        print("\nShutting down Flask server...")
        proc.terminate()
        proc.wait(timeout=5)
        print("Server stopped.")

if __name__ == "__main__":
    test()
