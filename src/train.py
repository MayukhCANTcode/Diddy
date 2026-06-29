"""
=========================================================
File Name : train.py

Purpose:
    Train the ResNet50 model on the EuroSAT dataset.

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
import torch.optim as optim

# Import model
from model import model

# Import dataloaders
from dataloader import train_loader, val_loader

# -------------------------------------------------------
# Device Configuration
# -------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)

print("=" * 60)
print("TRAINING SETUP")
print("=" * 60)

print(f"Device : {device}")

# -------------------------------------------------------
# Loss Function
# -------------------------------------------------------

criterion = nn.CrossEntropyLoss()

# -------------------------------------------------------
# Optimizer
# -------------------------------------------------------

optimizer = optim.Adam(
    model.fc.parameters(),
    lr=0.001
)

# -------------------------------------------------------
# Hyperparameters
# -------------------------------------------------------

NUM_EPOCHS = 10

print(f"Epochs : {NUM_EPOCHS}")
print(f"Learning Rate : 0.001")
print(f"Training Batches : {len(train_loader)}")
print(f"Validation Batches : {len(val_loader)}")

print("\nTraining setup completed successfully!")

# -------------------------------------------------------
# Store Best Validation Accuracy
# -------------------------------------------------------

best_validation_accuracy = 0.0

# -------------------------------------------------------
# Lists for plotting later
# -------------------------------------------------------

train_losses = []
validation_losses = []
validation_accuracies = []

# =======================================================
# TRAINING LOOP
# =======================================================

for epoch in range(NUM_EPOCHS):

    print("\n" + "=" * 60)
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    print("=" * 60)

    # ---------------------------------------------
    # TRAINING
    # ---------------------------------------------

    model.train()

    running_loss = 0.0

    for batch_index, (images, labels) in enumerate(train_loader):

        # Move batch to GPU
        images = images.to(device)
        labels = labels.to(device)

        # Forward Pass
        outputs = model(images)

        # Calculate Loss
        loss = criterion(outputs, labels)

        # Clear Previous Gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update Weights
        optimizer.step()

        running_loss += loss.item()

        # Print every 50 batches
        if (batch_index + 1) % 50 == 0:

            print(
                f"Batch [{batch_index + 1}/{len(train_loader)}] "
                f"Loss : {loss.item():.4f}"
            )

    # Average Training Loss
    epoch_loss = running_loss / len(train_loader)

    train_losses.append(epoch_loss)

    # ---------------------------------------------
    # VALIDATION
    # ---------------------------------------------

    model.eval()

    validation_loss = 0.0

    correct_predictions = 0

    total_images = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            validation_loss += loss.item()

            # Predicted Class
            _, predicted = torch.max(outputs, 1)

            correct_predictions += (predicted == labels).sum().item()

            total_images += labels.size(0)

    average_validation_loss = validation_loss / len(val_loader)

    validation_accuracy = (
        correct_predictions / total_images
    ) * 100

    validation_losses.append(average_validation_loss)

    validation_accuracies.append(validation_accuracy)

    # ---------------------------------------------
    # Print Results
    # ---------------------------------------------

    print(f"\nTraining Loss   : {epoch_loss:.4f}")
    print(f"Validation Loss : {average_validation_loss:.4f}")
    print(f"Validation Accuracy : {validation_accuracy:.2f}%")

    # ---------------------------------------------
    # Save Best Model
    # ---------------------------------------------

    if validation_accuracy > best_validation_accuracy:

        best_validation_accuracy = validation_accuracy

        torch.save(
            model.state_dict(),
            "models/checkpoints/best_model.pth"
        )

        print("✅ New Best Model Saved!")

# =======================================================
# TRAINING COMPLETE
# =======================================================

print("\n" + "=" * 60)
print("TRAINING FINISHED")
print("=" * 60)

print(f"Best Validation Accuracy : {best_validation_accuracy:.2f}%")
print("Best model saved to:")
print("models/checkpoints/best_model.pth")
