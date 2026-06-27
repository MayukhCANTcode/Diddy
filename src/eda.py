# ============================================================
# File: eda.py
# Purpose:
# Perform Exploratory Data Analysis (EDA) on the EuroSAT dataset.
# This helps us understand the dataset before training any model.
# ============================================================

# ----------------------------
# Import Required Libraries
# ----------------------------

# ImageFolder automatically reads images arranged in folders.
# Each folder name becomes a class label.
from torchvision.datasets import ImageFolder

# Used for plotting graphs and displaying images.
import matplotlib.pyplot as plt

# Counts the frequency of labels (images per class).
from collections import Counter

# Used to randomly select sample images.
import random


# ------------------------------------------------------------
# STEP 1 : Load the Dataset
# ------------------------------------------------------------
# The dataset is stored inside:
#
# data/
#   raw/
#      EuroSAT/
#          Forest/
#          River/
#          Highway/
#          ...
#
# ImageFolder automatically assigns:
#
# Forest -> 0
# Highway -> 1
# River -> 2
# ...
#
# based on folder names.
# ------------------------------------------------------------

dataset = ImageFolder("data/raw/EuroSat")


# ------------------------------------------------------------
# STEP 2 : Basic Dataset Information
# ------------------------------------------------------------

print("=" * 50)
print("EUROSAT DATASET INFORMATION")
print("=" * 50)

# Total number of images in the dataset
print(f"\nTotal Images : {len(dataset)}")

# Number of different land cover classes
print(f"Number of Classes : {len(dataset.classes)}")


# ------------------------------------------------------------
# STEP 3 : Display Class Names
# ------------------------------------------------------------

print("\nClass Names:")

for class_name in dataset.classes:
    print(f"- {class_name}")


# ------------------------------------------------------------
# STEP 4 : Display Label Mapping
# ------------------------------------------------------------
# Machine learning models cannot understand words.
# They understand numbers.
#
# Example:
#
# Forest -> 1
# River -> 8
#
# This mapping is automatically created by ImageFolder.
# ------------------------------------------------------------

print("\nClass Label Mapping:")

for class_name, label in dataset.class_to_idx.items():
    print(f"{class_name} --> {label}")


# ------------------------------------------------------------
# STEP 5 : Count Images in Each Class
# ------------------------------------------------------------
# dataset contains tuples:
#
# (image, label)
#
# We extract only the labels and count them.
# ------------------------------------------------------------

labels = [label for _, label in dataset]

counts = Counter(labels)

print("\nImages Per Class:")

for label, count in counts.items():
    print(f"{dataset.classes[label]} : {count}")


# ------------------------------------------------------------
# STEP 6 : Display Random Sample Images
# ------------------------------------------------------------
# We randomly pick 10 images from the dataset.
#
# This helps us visually inspect:
# • Image quality
# • Different land cover types
# • Any incorrect images
# ------------------------------------------------------------

fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for ax in axes.flatten():

    # Pick one random image
    image, label = random.choice(dataset)

    # Display image
    ax.imshow(image)

    # Display its class name
    ax.set_title(dataset.classes[label])

    # Hide axis numbers
    ax.axis("off")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# STEP 7 : Plot Class Distribution
# ------------------------------------------------------------
# A balanced dataset generally leads to better model training.
#
# This graph shows how many images belong to each class.
# ------------------------------------------------------------

class_names = [dataset.classes[label] for label in counts.keys()]
image_counts = list(counts.values())

plt.figure(figsize=(12, 5))

plt.bar(class_names, image_counts)

plt.title("EuroSAT Class Distribution")

plt.xlabel("Land Cover Class")

plt.ylabel("Number of Images")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# END OF EDA
# ------------------------------------------------------------
# At this point we know:
#
# ✔ Total number of images
# ✔ Number of classes
# ✔ Class names
# ✔ Label mapping
# ✔ Images per class
# ✔ Sample satellite images
# ✔ Class distribution
#
# We are now ready to preprocess the images and
# create DataLoaders in the next step.
# ------------------------------------------------------------
