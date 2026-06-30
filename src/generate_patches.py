"""
=========================================================
File Name : generate_patches.py

Purpose:
    Generate 64x64 image patches from Sentinel-2 RGB GeoTIFFs.

Author : Mayukh Das

Project:
    Deforestation Detection using Deep Learning
=========================================================
"""

# =====================================================
# Imports
# =====================================================

import os
import csv

import cv2
import rasterio
import numpy as np

# =====================================================
# Configuration
# =====================================================

INPUT_FOLDER = "data/processed"

OUTPUT_FOLDER = "data/patches"

METADATA_FOLDER = "data/metadata"

PATCH_SIZE = 64

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(METADATA_FOLDER, exist_ok=True)

# =====================================================
# Function
# =====================================================


def generate_patches(image_name):

    input_path = os.path.join(
        INPUT_FOLDER,
        image_name + ".tif"
    )

    output_dir = os.path.join(
        OUTPUT_FOLDER,
        image_name
    )

    os.makedirs(output_dir, exist_ok=True)

    metadata_path = os.path.join(
        METADATA_FOLDER,
        image_name + "_patches.csv"
    )

    print("=" * 60)
    print(f"Processing {image_name}")
    print("=" * 60)

    with rasterio.open(input_path) as src:

        image = src.read()

    image = np.transpose(
        image,
        (1, 2, 0)
    )

    height, width, _ = image.shape

    print(f"Image Size : {width} x {height}")

    patch_count = 0

    with open(
        metadata_path,
        "w",
        newline=""
    ) as csvfile:

        writer = csv.writer(csvfile)

        writer.writerow(
            [
                "patch_name",
                "x",
                "y"
            ]
        )

        for y in range(
            0,
            height - PATCH_SIZE + 1,
            PATCH_SIZE
        ):

            for x in range(
                0,
                width - PATCH_SIZE + 1,
                PATCH_SIZE
            ):

                patch = image[
                    y:y + PATCH_SIZE,
                    x:x + PATCH_SIZE
                ]

                patch_name = f"patch_{patch_count:06d}.png"

                save_path = os.path.join(
                    output_dir,
                    patch_name
                )

                cv2.imwrite(
                    save_path,
                    cv2.cvtColor(
                        patch,
                        cv2.COLOR_RGB2BGR
                    )
                )

                writer.writerow(
                    [
                        patch_name,
                        x,
                        y
                    ]
                )

                patch_count += 1

    print(f"Total Patches : {patch_count}")

    print(f"Saved to : {output_dir}")

    print(f"Metadata : {metadata_path}")

# =====================================================
# Main
# =====================================================


generate_patches("2018_rgb")

generate_patches("2024_rgb")

print("\n" + "=" * 60)

print("PATCH GENERATION COMPLETE")

print("=" * 60)
