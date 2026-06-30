"""
=========================================================
File Name : preprocess_sentinel.py

Purpose:
    Extract RGB images from Sentinel-2 SAFE folders.

Author : Mayukh Das

Project:
    Deforestation Detection using Deep Learning
=========================================================
"""

# =====================================================
# Import Libraries
# =====================================================

import os
from glob import glob

import cv2
import numpy as np
import rasterio

# =====================================================
# Paths
# =====================================================

SENTINEL_FOLDER = r"C:\S2"

OUTPUT_FOLDER = "data/processed"

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

# =====================================================
# Find SAFE folders
# =====================================================

safe_folders = glob(
    os.path.join(
        SENTINEL_FOLDER,
        "*.SAFE"
    )
)

print("=" * 60)
print("FOUND SENTINEL PRODUCTS")
print("=" * 60)

for folder in safe_folders:
    print(folder)

print()

# =====================================================
# Process each SAFE product
# =====================================================

for safe_folder in safe_folders:

    print("=" * 60)
    print(f"Processing : {os.path.basename(safe_folder)}")
    print("=" * 60)

    # -------------------------------------------------
    # Find R10m Folder
    # -------------------------------------------------

    matches = glob(
        os.path.join(
            safe_folder,
            "**",
            "R10m"
        ),
        recursive=True
    )

    if len(matches) == 0:
        print("ERROR: Could not find R10m folder")
        continue

    r10_folder = matches[0]

    print("R10m folder:")
    print(r10_folder)

    # -------------------------------------------------
    # Locate RGB Bands
    # -------------------------------------------------

    red = glob(
        os.path.join(
            r10_folder,
            "*B04_10m.jp2"
        )
    )

    green = glob(
        os.path.join(
            r10_folder,
            "*B03_10m.jp2"
        )
    )

    blue = glob(
        os.path.join(
            r10_folder,
            "*B02_10m.jp2"
        )
    )

    if not red or not green or not blue:
        print("RGB bands not found")
        continue

    red_path = red[0]
    green_path = green[0]
    blue_path = blue[0]
    # -------------------------------------------------
    # Read Bands
    # -------------------------------------------------

    with rasterio.open(red_path) as src:
        red = src.read(1)
        profile = src.profile

    with rasterio.open(green_path) as src:
        green = src.read(1)

    with rasterio.open(blue_path) as src:
        blue = src.read(1)

    # -------------------------------------------------
    # Stack RGB
    # -------------------------------------------------

    rgb = np.dstack(
        (
            red,
            green,
            blue
        )
    )

    print(f"RGB Shape : {rgb.shape}")

    # -------------------------------------------------
    # Normalize
    # -------------------------------------------------

    rgb = rgb.astype(np.float32)

    rgb /= 3000.0

    rgb = np.clip(
        rgb,
        0,
        1
    )

    rgb_8bit = (
        rgb * 255
    ).astype(np.uint8)

    # -------------------------------------------------
    # Determine Output Name
    # -------------------------------------------------

    folder_name = os.path.basename(safe_folder)

    if "2018" in folder_name:

        output_name = "2018_rgb"

    elif "2024" in folder_name:

        output_name = "2024_rgb"

    else:

        output_name = folder_name

    # -------------------------------------------------
    # Save PNG
    # -------------------------------------------------

    png_path = os.path.join(
        OUTPUT_FOLDER,
        output_name + ".png"
    )

    cv2.imwrite(
        png_path,
        cv2.cvtColor(
            rgb_8bit,
            cv2.COLOR_RGB2BGR
        )
    )

    print(f"Saved PNG : {png_path}")

    # -------------------------------------------------
    # Save GeoTIFF
    # -------------------------------------------------

    tif_path = os.path.join(
        OUTPUT_FOLDER,
        output_name + ".tif"
    )

    profile.update(
        driver="GTiff",
        count=3,
        dtype=rasterio.uint8
    )

    with rasterio.open(
        tif_path,
        "w",
        **profile
    ) as dst:

        dst.write(
            rgb_8bit[:, :, 0],
            1
        )

        dst.write(
            rgb_8bit[:, :, 1],
            2
        )

        dst.write(
            rgb_8bit[:, :, 2],
            3
        )

    print(f"Saved TIFF : {tif_path}")

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)
