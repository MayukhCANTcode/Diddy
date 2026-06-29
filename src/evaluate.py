"""
=========================================================
File Name : evaluate.py

Purpose:
    Evaluate the trained ResNet50 model on the EuroSAT
    test dataset.

Author : Mayukh Das

Project:
    Deforestation Detection using Deep Learning
=========================================================
"""

# -------------------------------------------------------
# Import Libraries
# -------------------------------------------------------

import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

# Import Model
from model import model

# Import Test DataLoader
from dataloader import test_loader

# -------------------------------------------------------
# EuroSAT Class Names
# -------------------------------------------------------

class_names = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake"
]

# -------------------------------------------------------
# Device Configuration
# -------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------
# Load Trained Model
# -------------------------------------------------------

model.load_state_dict(
    torch.load(
        "models/checkpoints/best_model.pth",
        map_location=device
    )
)

model = model.to(device)

print("=" * 60)
print("TRAINED MODEL LOADED")
print("=" * 60)

# -------------------------------------------------------
# Evaluation Mode
# -------------------------------------------------------

model.eval()

# -------------------------------------------------------
# Store Predictions
# -------------------------------------------------------

all_predictions = []

all_labels = []

# -------------------------------------------------------
# Disable Gradient Calculation
# -------------------------------------------------------

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Forward Pass
        outputs = model(images)

        # Get Predicted Class
        _, predictions = torch.max(outputs, 1)

        # Store Results
        all_predictions.extend(predictions.cpu().numpy())

        all_labels.extend(labels.cpu().numpy())

print("Evaluation Completed Successfully!")

# -------------------------------------------------------
# Calculate Test Accuracy
# -------------------------------------------------------

test_accuracy = accuracy_score(
    all_labels,
    all_predictions
)

print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"Test Accuracy : {test_accuracy * 100:.2f}%")

print("\nEvaluation Completed Successfully!")

# -------------------------------------------------------
# Classification Report
# -------------------------------------------------------

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    all_labels,
    all_predictions,
    target_names=class_names
)

print(report)

# -------------------------------------------------------
# Confusion Matrix
# -------------------------------------------------------

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\nConfusion Matrix Generated Successfully!")

# -------------------------------------------------------
# Plot Confusion Matrix
# -------------------------------------------------------

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("EuroSAT Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "outputs/figures/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
