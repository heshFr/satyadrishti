"""
Downloads ASVspoof 2019 LA dataset from Edinburgh DataShare and extracts it.
Source: https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip
"""
import os
import sys
import zipfile
import urllib.request
import tempfile
import time

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets", "asvspoof")
LA_URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip"


def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    if duration == 0:
        return
    progress_size = int(count * block_size)
    speed = progress_size / (1024 * 1024 * duration)
    percent = min(100, int(count * block_size * 100 / total_size))
    sys.stdout.write(f"\r  Progress: {percent}% | {progress_size / (1024*1024):.0f} MB | {speed:.1f} MB/s | {duration:.0f}s elapsed")
    sys.stdout.flush()


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "LA.zip")
    extract_dir = os.path.join(DATA_DIR, "LA")

    # Skip if already extracted
    if os.path.isdir(extract_dir) and os.listdir(extract_dir):
        print(f"LA directory already exists at {extract_dir}, skipping.")
        return

    # Download
    if not os.path.isfile(zip_path):
        print(f"Downloading ASVspoof 2019 LA from Edinburgh DataShare...")
        print(f"  URL: {LA_URL}")
        print(f"  Destination: {zip_path}")
        fh, tmp_path = tempfile.mkstemp(dir=DATA_DIR, suffix=".zip.tmp")
        f = os.fdopen(fh, 'w')
        f.close()
        try:
            urllib.request.urlretrieve(LA_URL, tmp_path, reporthook=reporthook)
            print()
            os.rename(tmp_path, zip_path)
            print("Download complete.")
        except Exception as e:
            print(f"\nDownload failed: {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return
    else:
        print(f"ZIP already exists: {zip_path}")

    # Extract
    print(f"Extracting {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(DATA_DIR)
        print("Extraction complete.")
        print(f"  Output: {extract_dir}")
    except Exception as e:
        print(f"Extraction failed: {e}")


if __name__ == "__main__":
    main()
