"""
Satya Drishti
=============
Downloads the ASVspoof 2021 LA and DF datasets for audio deepfake detection.
Warning: This dataset is extremely large (~50GB+). Ensure sufficient disk space.
"""

import os
import tarfile
import requests
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets", "asvspoof")

# The Zenodo records for ASVspoof 2021 Data
ZENODO_BASE_URL = "https://zenodo.org/record/4837263/files/"
FILES = [
    "ASVspoof2021_LA_eval.tar.gz",
    "ASVspoof2021_DF_eval.tar.gz"
]

def download_file(url, out_path):
    if os.path.exists(out_path):
        print(f"File {os.path.basename(out_path)} already exists, skipping download.")
        return

    print(f"Downloading from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024 # 1 Megabyte
        
        with tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True) as progress_bar:
            with open(out_path, 'wb') as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
    except Exception as e:
        print(f"Download failed: {e}")
        if os.path.exists(out_path):
            os.remove(out_path)

def extract_tarball(tar_path, extract_path):
    print(f"Extracting {os.path.basename(tar_path)}...")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            
            # Extract with progress
            members = tar.getmembers()
            for member in tqdm(members, desc="Extracting"):
                tar.extract(member, path=extract_path)
        print("Extraction complete.")
    except Exception as e:
        print(f"Extraction failed: {e}")

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    for filename in FILES:
        url = ZENODO_BASE_URL + filename
        out_path = os.path.join(DATA_DIR, filename)
        
        # 1. Download
        download_file(url, out_path)
        
        # 2. Extract
        extract_tarball(out_path, DATA_DIR)
        
        print(f"Successfully processed {filename}.")
        
    print("\nASVspoof 2021 download sequence initiated. Proceed to parse protocols to build the DataLoader.")

if __name__ == "__main__":
    main()
