"""Download CAMUS patients from the public CREATIS server.

    python scripts/download_camus.py --out data/raw            # all 500 patients, about 3.5 GB
    python scripts/download_camus.py --out data/raw --n 150    # a subset for a quick run
"""
import argparse
import http.client
import io
import json
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

API = "https://humanheart-project.creatis.insa-lyon.fr/database/api/v1"
NIFTI_FOLDER = "63fde55f73e9f004868fb7ac"


def fetch(url, attempts=5):
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=300) as response:
                return response.read()
        except (OSError, http.client.HTTPException):  # the server drops connections under load
            if attempt == attempts - 1:
                raise
            time.sleep(2 * (attempt + 1))


def download_patient(folder, out_dir):
    if (out_dir / folder["name"]).exists():
        return
    archive = zipfile.ZipFile(io.BytesIO(fetch(f"{API}/folder/{folder['_id']}/download")))
    archive.extractall(out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw")
    parser.add_argument("--n", type=int, default=0, help="number of patients, 0 for all")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    listing = f"{API}/folder?parentType=folder&parentId={NIFTI_FOLDER}&limit={args.n}&sort=name"
    folders = json.loads(fetch(listing))
    with ThreadPoolExecutor(4) as pool:
        for done, _ in enumerate(pool.map(lambda f: download_patient(f, out_dir), folders), 1):
            if done % 25 == 0 or done == len(folders):
                print(f"{done}/{len(folders)} patients")
