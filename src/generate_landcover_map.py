"""
=========================================================
File Name : generate_landcover_map.py

Purpose:
    Generate land-cover maps from prediction CSV files.

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
import numpy as np

from pathlib import Path

# =====================================================
# Project Paths
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PREDICTION_FOLDER = PROJECT_ROOT / "data" / "predictions"

OUTPUT_FOLDER = PROJECT_ROOT / "outputs"

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

PATCH_SIZE = 64

# =====================================================
# Class Colors (RGB)
# =====================================================

CLASS_COLORS = {

    "AnnualCrop": (255, 255, 0),

    "Forest": (34, 139, 34),

    "HerbaceousVegetation": (124, 252, 0),

    "Highway": (128, 128, 128),

    "Industrial": (128, 0, 128),

    "Pasture": (50, 205, 50),

    "PermanentCrop": (210, 180, 140),

    "Residential": (220, 20, 60),

    "River": (30, 144, 255),

    "SeaLake": (0, 0, 139)

}

# =====================================================
# Generate Land Cover Map
# =====================================================


def generate_landcover_map(file_name):

    print("\n" + "=" * 60)
    print(f"Generating Map : {file_name}")
    print("=" * 60)

    csv_path = PREDICTION_FOLDER / f"{file_name}_predictions.csv"

    predictions = []

    max_x = 0
    max_y = 0

    # ----------------------------------------------
    # Read Prediction CSV
    # ----------------------------------------------

    with open(csv_path, "r") as file:

        reader = csv.DictReader(file)

        for row in reader:

            x = int(row["x"])
            y = int(row["y"])

            predicted_class = row["predicted_class"]

            predictions.append(
                (
                    x,
                    y,
                    predicted_class
                )
            )

            max_x = max(max_x, x)
            max_y = max(max_y, y)

    # ----------------------------------------------
    # Create Blank Canvas
    # ----------------------------------------------

    width = max_x + PATCH_SIZE
    height = max_y + PATCH_SIZE

    landcover = np.zeros(
        (
            height,
            width,
            3
        ),
        dtype=np.uint8
    )

    print(f"Canvas Size : {width} x {height}")

    # ----------------------------------------------
    # Paint Every Patch
    # ----------------------------------------------

    for x, y, predicted_class in predictions:

        color = CLASS_COLORS[predicted_class]

        landcover[
            y:y + PATCH_SIZE,
            x:x + PATCH_SIZE
        ] = color

    print(f"Total Patches Drawn : {len(predictions)}")

    # ----------------------------------------------
    # Save PNG
    # ----------------------------------------------

    png_path = OUTPUT_FOLDER / f"{file_name}_landcover.png"

    cv2.imwrite(

        str(png_path),

        cv2.cvtColor(
            landcover,
            cv2.COLOR_RGB2BGR
        )

    )

    print(f"Saved : {png_path}")

    return landcover

# =====================================================
# Main Function
# =====================================================


def main():

    print("=" * 60)
    print("GENERATING LAND COVER MAPS")
    print("=" * 60)

    generate_landcover_map("2018_rgb")

    generate_landcover_map("2024_rgb")

    print("\n" + "=" * 60)
    print("LAND COVER MAP GENERATION COMPLETE")
    print("=" * 60)

    print("\nGenerated Files:")

    print(OUTPUT_FOLDER / "2018_rgb_landcover.png")
    print(OUTPUT_FOLDER / "2024_rgb_landcover.png")


# =====================================================
# Entry Point
# =====================================================

if __name__ == "__main__":

    main()
