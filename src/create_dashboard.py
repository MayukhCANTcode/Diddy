"""
=========================================================
File Name : create_dashboard.py

Purpose:
    Create a dashboard summarizing the entire
    deforestation detection workflow.

Author : Mayukh Das
=========================================================
"""

from pathlib import Path
import cv2
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT = PROJECT_ROOT / "outputs"
PROCESSED = PROJECT_ROOT / "data" / "processed"

# ----------------------------------------------------
# Load Images
# ----------------------------------------------------

rgb_2018 = cv2.imread(
    str(PROCESSED / "2018_rgb.png")
)

rgb_2024 = cv2.imread(
    str(PROCESSED / "2024_rgb.png")
)

landcover_2018 = cv2.imread(
    str(OUTPUT / "2018_rgb_landcover.png")
)

landcover_2024 = cv2.imread(
    str(OUTPUT / "2024_rgb_landcover.png")
)

overlay = cv2.imread(
    str(OUTPUT / "forest_loss_overlay.png")
)

# Convert BGR → RGB

rgb_2018 = cv2.cvtColor(rgb_2018, cv2.COLOR_BGR2RGB)
rgb_2024 = cv2.cvtColor(rgb_2024, cv2.COLOR_BGR2RGB)
landcover_2018 = cv2.cvtColor(landcover_2018, cv2.COLOR_BGR2RGB)
landcover_2024 = cv2.cvtColor(landcover_2024, cv2.COLOR_BGR2RGB)
overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

# ----------------------------------------------------
# Dashboard
# ----------------------------------------------------

fig = plt.figure(figsize=(18, 12))

fig.suptitle(
    "Deforestation Detection using Deep Learning and Sentinel-2",
    fontsize=18,
    fontweight="bold"
)

# ------------------------

plt.subplot(2, 3, 1)
plt.imshow(rgb_2018)
plt.title("2018 Sentinel RGB")
plt.axis("off")

# ------------------------

plt.subplot(2, 3, 2)
plt.imshow(rgb_2024)
plt.title("2024 Sentinel RGB")
plt.axis("off")

# ------------------------

plt.subplot(2, 3, 3)
plt.imshow(overlay)
plt.title("Forest Loss Overlay")
plt.axis("off")

# ------------------------

plt.subplot(2, 3, 4)
plt.imshow(landcover_2018)
plt.title("2018 Land Cover")
plt.axis("off")

# ------------------------

plt.subplot(2, 3, 5)
plt.imshow(landcover_2024)
plt.title("2024 Land Cover")
plt.axis("off")

# ------------------------

plt.subplot(2, 3, 6)

plt.axis("off")

plt.text(

    0,

    1,

    """
PROJECT SUMMARY

Study Area:
Novo Progresso, Brazil

Satellite:
Sentinel-2

Model:
ResNet50 Transfer Learning

Validation Accuracy:
94%

Test Accuracy:
94.33%

Forest Patches (2018):
15088

Forest Loss:
2923

Forest Loss:
19.37%
""",

    fontsize=13,

    verticalalignment="top"

)

plt.tight_layout()

save_path = OUTPUT / "final_dashboard.png"

plt.savefig(

    save_path,

    dpi=300,

    bbox_inches="tight"

)

plt.close()

print("="*60)
print("FINAL DASHBOARD GENERATED")
print("="*60)
print(save_path)
