import os
import requests
import sys

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

FILES_TO_DOWNLOAD = {
    "df_yield_merged.csv": [
        "https://zenodo.org/records/18761663/files/df_yield_merged.csv?download=1",
        "https://raw.githubusercontent.com/gnatnib/crop_yield_prediction/master/yield_df.csv",
    ],
    "yield.csv": [
        "https://raw.githubusercontent.com/gnatnib/crop_yield_prediction/master/yield.csv",
        "https://zenodo.org/records/18761663/files/yield.csv?download=1",
    ],
    "rainfall.csv": [
        "https://raw.githubusercontent.com/gnatnib/crop_yield_prediction/master/rainfall.csv",
        "https://zenodo.org/records/18761663/files/rainfall.csv?download=1",
    ],
    "pesticides.csv": [
        "https://raw.githubusercontent.com/gnatnib/crop_yield_prediction/master/pesticides.csv",
        "https://zenodo.org/records/18761663/files/pesticides.csv?download=1",
    ],
    "temp.csv": [
        "https://raw.githubusercontent.com/gnatnib/crop_yield_prediction/master/temp.csv",
        "https://zenodo.org/records/18761663/files/temp.csv?download=1",
    ],
}

def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    for fname, urls in FILES_TO_DOWNLOAD.items():
        dest = os.path.join(DATA_DIR, fname)
        if os.path.exists(dest) and os.path.getsize(dest) > 1024:
            print(f"File already exists: {fname} ({os.path.getsize(dest):,} bytes)")
            continue

        print(f"Downloading {fname}...")
        downloaded = False
        for url in urls:
            try:
                print(f"  Trying {url}...")
                with requests.get(url, headers=headers, stream=True, timeout=30) as r:
                    r.raise_for_status()
                    with open(dest, "wb") as f:
                        for chunk in r.iter_content(chunk_size=65536):
                            if chunk:
                                f.write(chunk)
                if os.path.exists(dest) and os.path.getsize(dest) > 1024:
                    print(f"  Successfully downloaded {fname} ({os.path.getsize(dest):,} bytes)")
                    downloaded = True
                    break
            except Exception as e:
                print(f"  Note: {url} failed: {e}")

        if not downloaded:
            print(f"Warning: Could not download {fname}")

if __name__ == "__main__":
    download_dataset()
