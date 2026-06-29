"""
=========================================================
File Name : model.py

Purpose:
    Load a pretrained ResNet50 model and modify it
    for EuroSAT land-cover classification.

Author : Mayukh Das

Project:
    Deforestation Detection using Deep Learning
=========================================================
"""

# -------------------------------------------------------
# Import Libraries
# -------------------------------------------------------

import torch
import torch.nn as nn
from torchvision import models

# -------------------------------------------------------
# Number of Output Classes
# -------------------------------------------------------

NUM_CLASSES = 10

# -------------------------------------------------------
# Load Pretrained ResNet50
# -------------------------------------------------------

# weights="DEFAULT" downloads pretrained ImageNet weights
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

print("=" * 60)
print("PRETRAINED RESNET50 LOADED")
print("=" * 60)

# -------------------------------------------------------
# Freeze All Layers
# -------------------------------------------------------
# We don't want to retrain the entire network.
# We only want to train the final classifier.
# -------------------------------------------------------

for parameter in model.parameters():
    parameter.requires_grad = False

# -------------------------------------------------------
# Replace the Final Fully Connected Layer
# -------------------------------------------------------

num_features = model.fc.in_features

model.fc = nn.Linear(
    in_features=num_features,
    out_features=NUM_CLASSES
)

# Allow training of the new classifier
for parameter in model.fc.parameters():
    parameter.requires_grad = True

print("\nFinal Layer Replaced Successfully!")

print(f"\nInput Features : {num_features}")
print(f"Output Classes : {NUM_CLASSES}")

# -------------------------------------------------------
# Select Device
# -------------------------------------------------------
# If a CUDA-compatible GPU is available, use it.
# Otherwise, fall back to the CPU.
# -------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n" + "=" * 60)
print("DEVICE INFORMATION")
print("=" * 60)

print(f"Selected Device : {device}")

# -------------------------------------------------------
# Move Model to Device
# -------------------------------------------------------

model = model.to(device)

print("Model successfully moved to the selected device!")

# -------------------------------------------------------
# GPU Information
# -------------------------------------------------------

if device.type == "cuda":
    print(f"GPU Name : {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version : {torch.version.cuda}")
else:
    print("Running on CPU.")
