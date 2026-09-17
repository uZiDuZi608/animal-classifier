"""
src/download_data.py
--------------------
Downloads the public UCI Zoo dataset (zoo.data and zoo.names)
and saves raw files into data/raw/ with complete provenance metadata.

Educational Focus:
- Machine learning begins with data acquisition from legitimate, verifiable sources.
- We never fabricate data.
- We document data provenance, licensing, and schema definitions.
"""

import os
import sys
import json
import urllib.request
import hashlib
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data"
NAMES_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.names"

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")

def compute_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def download_file(url: str, dest_path: str) -> None:
    print(f"[*] Downloading: {url}")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    with urllib.request.urlopen(req) as response:
        content = response.read()
    with open(dest_path, "wb") as f:
        f.write(content)
    print(f"[+] Saved to: {dest_path} ({len(content)} bytes)")

def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    data_dest = os.path.join(RAW_DIR, "zoo.data")
    names_dest = os.path.join(RAW_DIR, "zoo.names")

    download_file(DATA_URL, data_dest)
    download_file(NAMES_URL, names_dest)

    data_hash = compute_sha256(data_dest)
    names_hash = compute_sha256(names_dest)

    metadata = {
        "dataset_name": "Zoo Database",
        "creator": "Richard S. Forsyth",
        "date_donated": "1990-05-15",
        "source_url_data": DATA_URL,
        "source_url_names": NAMES_URL,
        "repository": "UCI Machine Learning Repository",
        "citation": "Forsyth, R. (1990). Zoo. UCI Machine Learning Repository. https://doi.org/10.24432/C5659V",
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "downloaded_at": datetime.now().isoformat(),
        "sha256_zoo_data": data_hash,
        "sha256_zoo_names": names_hash
    }

    metadata_path = os.path.join(RAW_DIR, "dataset_info.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"[+] Provenance metadata recorded at: {metadata_path}")
    print("[OK] Data acquisition completed successfully.")

if __name__ == "__main__":
    main()
