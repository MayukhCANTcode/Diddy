"""
=========================================================
File Name : create_comparison.py

Purpose:
    Create a side-by-side comparison of the
    2018 and 2024 land-cover maps.

Author : Mayukh Das
=========================================================
"""

from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_FOLDER = PROJECT_ROOT / "outputs"

image_2018 = cv2.imread(
    str(
        OUTPUT_FOLDER /
        "2018_rgb_landcover.png"
    )
)

image_2024 = cv2.imread(
    str(
        OUTPUT_FOLDER /
        "2024_rgb_landcover.png"
    )
)

height = max(
    image_2018.shape[0],
    image_2024.shape[0]
)

width = image_2018.shape[1] + image_2024.shape[1]

comparison = np.ones(
    (
        height + 80,
        width,
        3
    ),
    dtype=np.uint8
) * 255

comparison[
    80:80 + image_2018.shape[0],
    :image_2018.shape[1]
] = image_2018

comparison[
    80:80 + image_2024.shape[0],
    image_2018.shape[1]:
] = image_2024

cv2.putText(

    comparison,

    "2018 Land Cover",

    (100, 50),

    cv2.FONT_HERSHEY_SIMPLEX,

    1,

    (0, 0, 0),

    2

)

cv2.putText(

    comparison,

    "2024 Land Cover",

    (

        image_2018.shape[1] + 100,

        50

    ),

    cv2.FONT_HERSHEY_SIMPLEX,

    1,

    (0, 0, 0),

    2

)

save_path = OUTPUT_FOLDER / "comparison.png"

cv2.imwrite(
    str(save_path),
    comparison
)

print("=" * 60)
print("COMPARISON IMAGE GENERATED")
print("=" * 60)
print(save_path)
