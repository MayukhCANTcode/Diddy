"""
=========================================================
File Name : dataloader.py

Purpose:
    Create PyTorch DataLoaders.

Author : Mayukh Das
=========================================================
"""

from torch.utils.data import DataLoader

from dataset import (
    train_dataset,
    val_dataset,
    test_dataset
)

BATCH_SIZE = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("=" * 50)
print("DATALOADERS CREATED")
print("=" * 50)

print(f"Training Batches   : {len(train_loader)}")
print(f"Validation Batches : {len(val_loader)}")
print(f"Testing Batches    : {len(test_loader)}")
