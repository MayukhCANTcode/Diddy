"""
Extract all Sentinel-2 SAFE ZIP files.
"""

import os
import zipfile
from glob import glob

ZIP_FOLDER = "sentinel_data"

zip_files = glob(os.path.join(ZIP_FOLDER, "*.zip"))

print("=" * 60)
print("FOUND ZIP FILES")
print("=" * 60)

for zip_path in zip_files:

    print(os.path.basename(zip_path))

    output_folder = os.path.splitext(zip_path)[0]

    os.makedirs(output_folder, exist_ok=True)

    print(f"Extracting to {output_folder}")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_folder)

    print("Done\n")

print("=" * 60)
print("ALL FILES EXTRACTED")
print("=" * 60)
