"""
=========================================================
File Name : detect_deforestation.py

Purpose:
    Detect deforestation by comparing the predicted
    land-cover classes of 2018 and 2024.

Author : Mayukh Das

Project:
    Deforestation Detection using Deep Learning
=========================================================
"""

# =====================================================
# Import Libraries
# =====================================================

import csv
from pathlib import Path

import cv2
import numpy as np

# =====================================================
# Project Paths
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PREDICTIONS_FOLDER = PROJECT_ROOT / "data" / "predictions"

OUTPUT_FOLDER = PROJECT_ROOT / "outputs"

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

PATCH_SIZE = 64

# =====================================================
# Read Prediction CSV
# =====================================================


def load_predictions(file_name):

    csv_path = (
        PREDICTIONS_FOLDER /
        f"{file_name}_predictions.csv"
    )

    predictions = {}

    max_x = 0
    max_y = 0

    with open(csv_path, "r") as file:

        reader = csv.DictReader(file)

        for row in reader:

            x = int(row["x"])
            y = int(row["y"])

            predictions[(x, y)] = {

                "class": row["predicted_class"],

                "confidence": float(
                    row["confidence"]
                )

            }

            max_x = max(max_x, x)
            max_y = max(max_y, y)

    width = max_x + PATCH_SIZE
    height = max_y + PATCH_SIZE

    return predictions, width, height

# =====================================================
# Detect Deforestation
# =====================================================


def detect_deforestation():

    print("\n" + "=" * 60)
    print("DETECTING DEFORESTATION")
    print("=" * 60)

    predictions_2018, width, height = load_predictions(
        "2018_rgb"
    )

    predictions_2024, _, _ = load_predictions(
        "2024_rgb"
    )

    # ---------------------------------------------
    # Create Output Images
    # ---------------------------------------------

    forest_loss_mask = np.zeros(
        (
            height,
            width,
            3
        ),
        dtype=np.uint8
    )

    transition_map = np.zeros(
        (
            height,
            width,
            3
        ),
        dtype=np.uint8
    )

    # ---------------------------------------------
    # Statistics
    # ---------------------------------------------

    total_forest_2018 = 0

    forest_loss = 0

    transition_counts = {}

    # ---------------------------------------------
    # Compare Every Patch
    # ---------------------------------------------

    for location in predictions_2018:

        class_2018 = predictions_2018[location]["class"]

        class_2024 = predictions_2024[location]["class"]

        x, y = location

        # Count total forest

        if class_2018 == "Forest":

            total_forest_2018 += 1

        # Detect Forest Loss

        if (

            class_2018 == "Forest"

            and

            class_2024 != "Forest"

        ):

            forest_loss += 1

            # -----------------------------
            # Forest Loss Mask
            # -----------------------------

            forest_loss_mask[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ] = (255, 0, 0)

            # -----------------------------
            # Transition Statistics
            # -----------------------------

            transition = (

                class_2018,

                class_2024

            )

            transition_counts[transition] = (

                transition_counts.get(

                    transition,

                    0

                )

                + 1

            )

            # -----------------------------
            # Transition Colors
            # -----------------------------

            if class_2024 == "AnnualCrop":

                color = (255, 255, 0)

            elif class_2024 == "Pasture":

                color = (50, 205, 50)

            elif class_2024 == "Residential":

                color = (220, 20, 60)

            elif class_2024 == "Industrial":

                color = (128, 0, 128)

            elif class_2024 == "Highway":

                color = (128, 128, 128)

            else:

                color = (255, 165, 0)

            transition_map[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ] = color

    print(f"Forest Patches (2018) : {total_forest_2018}")

    print(f"Forest Loss Patches   : {forest_loss}")

    return (

        forest_loss_mask,

        transition_map,

        transition_counts,

        total_forest_2018,

        forest_loss

    )

# =====================================================
# Save Results
# =====================================================


def save_results():

    (
        forest_loss_mask,
        transition_map,
        transition_counts,
        total_forest,
        forest_loss

    ) = detect_deforestation()

    # ---------------------------------------------
    # Save Images
    # ---------------------------------------------

    forest_loss_path = (
        OUTPUT_FOLDER /
        "forest_loss_mask.png"
    )

    transition_map_path = (
        OUTPUT_FOLDER /
        "deforestation_map.png"
    )

    cv2.imwrite(

        str(forest_loss_path),

        cv2.cvtColor(
            forest_loss_mask,
            cv2.COLOR_RGB2BGR
        )

    )

    cv2.imwrite(

        str(transition_map_path),

        cv2.cvtColor(
            transition_map,
            cv2.COLOR_RGB2BGR
        )

    )

    # ---------------------------------------------
    # Save Transition Matrix
    # ---------------------------------------------

    transition_csv = (
        OUTPUT_FOLDER /
        "transition_matrix.csv"
    )

    with open(

        transition_csv,

        "w",

        newline=""

    ) as file:

        writer = csv.writer(file)

        writer.writerow(

            [

                "From",

                "To",

                "Patch Count"

            ]

        )

        for transition, count in sorted(
            transition_counts.items()
        ):

            writer.writerow(

                [

                    transition[0],

                    transition[1],

                    count

                ]

            )

    # ---------------------------------------------
    # Save Statistics
    # ---------------------------------------------

    statistics_csv = (
        OUTPUT_FOLDER /
        "change_statistics.csv"
    )

    if total_forest > 0:

        forest_loss_percentage = (

            forest_loss /

            total_forest

        ) * 100

    else:

        forest_loss_percentage = 0

    with open(

        statistics_csv,

        "w",

        newline=""

    ) as file:

        writer = csv.writer(file)

        writer.writerow(

            [

                "Metric",

                "Value"

            ]

        )

        writer.writerow(

            [

                "Forest Patches (2018)",

                total_forest

            ]

        )

        writer.writerow(

            [

                "Forest Loss Patches",

                forest_loss

            ]

        )

        writer.writerow(

            [

                "Forest Loss (%)",

                round(

                    forest_loss_percentage,

                    2

                )

            ]

        )

    # ---------------------------------------------
    # Print Summary
    # ---------------------------------------------

    print("\n" + "=" * 60)

    print("SUMMARY")

    print("=" * 60)

    print(f"Forest Patches (2018) : {total_forest}")

    print(f"Forest Loss Patches   : {forest_loss}")

    print(f"Forest Loss (%)       : {forest_loss_percentage:.2f}%")

    print("\nOutputs Generated:")

    print(forest_loss_path)

    print(transition_map_path)

    print(transition_csv)

    print(statistics_csv)


# =====================================================
# Main
# =====================================================

def main():

    print("=" * 60)

    print("DAY 8 - DEFORESTATION DETECTION")

    print("=" * 60)

    save_results()

    print("\n" + "=" * 60)

    print("DEFORESTATION ANALYSIS COMPLETE")

    print("=" * 60)


# =====================================================
# Entry Point
# =====================================================

if __name__ == "__main__":

    main()
