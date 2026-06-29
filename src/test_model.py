"""
=========================================================
File Name : test_model.py

Purpose:
    Test whether the DataLoader and ResNet50
    work together correctly.

Author : Mayukh Das
=========================================================
"""

# -------------------------------------------------------
# Import Libraries
# -------------------------------------------------------

import torch

from model import model
from dataloader import train_loader

# -------------------------------------------------------
# Select Device
# -------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------
# Get One Batch
# -------------------------------------------------------

images, labels = next(iter(train_loader))

print("=" * 60)
print("BATCH INFORMATION")
print("=" * 60)

print(f"Image Batch Shape : {images.shape}")
print(f"Label Shape       : {labels.shape}")

# -------------------------------------------------------
# Move Batch to GPU
# -------------------------------------------------------

images = images.to(device)

# -------------------------------------------------------
# Forward Pass
# -------------------------------------------------------

outputs = model(images)

print("\n")

print("=" * 60)
print("MODEL OUTPUT")
print("=" * 60)

print(f"Output Shape : {outputs.shape}")
