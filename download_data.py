import os
import requests
import sys

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CSV_PATH = os.path.join(DATA_DIR, "df_yield_merged.csv")

ZENODO_RECORD_URL = "https://zenodo.org/api/records/18761663"
DIRECT_URLS = [
    "https://zenodo.org/records/18761663/files/df_yield_merged.csv?download=1",
    "https://zenodo.org/records/18761663/files/df_yield_merged.csv",
]

def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 1024:
        print(f"Dataset already exists at: {CSV_PATH} ({os.path.getsize(CSV_PATH):,} bytes)")
        return CSV_PATH

    print("Attempting to download dataset from Zenodo Record 18761663...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    # First attempt: Query Zenodo API to get the exact file download URL
    download_url = None
    try:
        print(f"Querying Zenodo API: {ZENODO_RECORD_URL}")
        res = requests.get(ZENODO_RECORD_URL, headers=headers, timeout=15)
        if res.status_code == 200:
            data = res.json()
            files = data.get("files", [])
            for f in files:
                fname = f.get("key", "") or f.get("filename", "")
                if "df_yield_merged.csv" in fname:
                    download_url = f.get("links", {}).get("self")
                    print(f"Found file in API: {fname} -> {download_url}")
                    break
    except Exception as e:
        print(f"API query note: {e}")

    urls_to_try = [download_url] if download_url else []
    urls_to_try.extend(DIRECT_URLS)

    for url in urls_to_try:
        if not url:
            continue
        try:
            print(f"Downloading from {url}...")
            with requests.get(url, headers=headers, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(CSV_PATH, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
            if os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 1024:
                print(f"Successfully downloaded to: {CSV_PATH} ({os.path.getsize(CSV_PATH):,} bytes)")
                return CSV_PATH
        except Exception as e:
            print(f"Failed from {url}: {e}")

    raise RuntimeError("Could not download df_yield_merged.csv from Zenodo.")

if __name__ == "__main__":
    download_dataset()
