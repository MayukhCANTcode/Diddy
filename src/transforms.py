"""
=========================================================
File Name : transforms.py

Purpose:
    Define all image preprocessing pipelines used in
    the project.

Author : Mayukh Das

Project:
    Deforestation Detection using Deep Learning
=========================================================
"""

# -------------------------------------------------------
# Import torchvision transforms
# -------------------------------------------------------

from torchvision import transforms


# -------------------------------------------------------
# ImageNet Mean and Standard Deviation
# -------------------------------------------------------
# ResNet50 was trained on ImageNet.
# To make our satellite images compatible with the
# pretrained model, we normalize using the same values.
# -------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# -------------------------------------------------------
# Training Transform
# -------------------------------------------------------
# Used only while training.
#
# Includes data augmentation to improve generalization.
# -------------------------------------------------------

train_transform = transforms.Compose([

    # Resize 64x64 images to 224x224
    transforms.Resize((224, 224)),

    # Randomly flip images horizontally
    transforms.RandomHorizontalFlip(p=0.5),

    # Randomly flip images vertically
    transforms.RandomVerticalFlip(p=0.5),

    # Convert PIL Image to PyTorch Tensor
    transforms.ToTensor(),

    # Normalize pixel values
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )

])


# -------------------------------------------------------
# Validation Transform
# -------------------------------------------------------
# No augmentation.
# We want to evaluate on original images.
# -------------------------------------------------------

val_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )

])


# -------------------------------------------------------
# Test Transform
# -------------------------------------------------------
# Exactly the same as validation.
# Never augment test images.
# -------------------------------------------------------

test_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )

])
