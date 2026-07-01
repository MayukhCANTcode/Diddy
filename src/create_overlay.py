"""
=========================================================
File Name : create_overlay.py

Purpose:
    Overlay detected forest loss on the original
    Sentinel RGB image.

Author : Mayukh Das
=========================================================
"""

from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED = PROJECT_ROOT / "data" / "processed"

OUTPUT = PROJECT_ROOT / "outputs"

rgb = cv2.imread(
    str(PROCESSED / "2024_rgb.png")
)

mask = cv2.imread(
    str(OUTPUT / "forest_loss_mask.png")
)

# Resize mask if necessary
if rgb.shape != mask.shape:

    mask = cv2.resize(
        mask,
        (
            rgb.shape[1],
            rgb.shape[0]
        ),
        interpolation=cv2.INTER_NEAREST
    )

overlay = rgb.copy()

alpha = 0.55

red = np.zeros_like(rgb)

red[:, :, 2] = 255

loss_pixels = mask[:, :, 2] > 0

overlay[loss_pixels] = cv2.addWeighted(

    rgb[loss_pixels],

    1 - alpha,

    red[loss_pixels],

    alpha,

    0

)

save_path = OUTPUT / "forest_loss_overlay.png"

cv2.imwrite(
    str(save_path),
    overlay
)

print("=" * 60)
print("FOREST LOSS OVERLAY GENERATED")
print("=" * 60)
print(save_path)
