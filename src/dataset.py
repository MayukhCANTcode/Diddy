"""
=========================================================
File Name : dataset.py

Purpose:
    Create train, validation and test datasets
    with the correct transforms.

Author : Mayukh Das
=========================================================
"""

from pathlib import Path
import torch

from torchvision.datasets import ImageFolder
from torch.utils.data import random_split

from transforms import (
    train_transform,
    val_transform,
    test_transform
)

# ---------------------------------------------------
# Project Path
# ---------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "EuroSat"

# ---------------------------------------------------
# Load Dataset WITHOUT transforms
# ---------------------------------------------------

full_dataset = ImageFolder(DATASET_PATH)

dataset_size = len(full_dataset)

train_size = int(dataset_size * 0.8)

val_size = int(dataset_size * 0.1)

test_size = dataset_size - train_size - val_size

generator = torch.Generator().manual_seed(42)

train_subset, val_subset, test_subset = random_split(
    full_dataset,
    [train_size, val_size, test_size],
    generator=generator
)

# ---------------------------------------------------
# Create datasets WITH transforms
# ---------------------------------------------------

train_dataset = ImageFolder(
    DATASET_PATH,
    transform=train_transform
)

val_dataset = ImageFolder(
    DATASET_PATH,
    transform=val_transform
)

test_dataset = ImageFolder(
    DATASET_PATH,
    transform=test_transform
)

# Apply the split indices

train_dataset.samples = [train_dataset.samples[i]
                         for i in train_subset.indices]

val_dataset.samples = [val_dataset.samples[i] for i in val_subset.indices]

test_dataset.samples = [test_dataset.samples[i] for i in test_subset.indices]

print("=" * 50)
print("DATASETS CREATED")
print("=" * 50)

print(f"Training Images   : {len(train_dataset)}")
print(f"Validation Images : {len(val_dataset)}")
print(f"Testing Images    : {len(test_dataset)}")
