"""
=========================================================
File Name : run_inference.py

Purpose:
    Run inference on Sentinel-2 image patches using the
    trained ResNet50 model.

Author : Mayukh Das

Project:
    Deforestation Detection using Deep Learning
=========================================================
"""

# =====================================================
# Import Libraries
# =====================================================

import os
import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from PIL import Image

from torchvision import transforms

from model import model

# =====================================================
# Device
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("DEVICE")
print("=" * 60)
print(device)

# =====================================================
# Paths
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PATCH_FOLDER = PROJECT_ROOT / "data" / "patches"

METADATA_FOLDER = PROJECT_ROOT / "data" / "metadata"

OUTPUT_FOLDER = PROJECT_ROOT / "data" / "predictions"

MODEL_PATH = PROJECT_ROOT / "models" / "checkpoints" / "best_model.pth"

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

# =====================================================
# Class Names
# =====================================================

CLASS_NAMES = [

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

# =====================================================
# Transform
# =====================================================

inference_transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[0.485, 0.456, 0.406],

        std=[0.229, 0.224, 0.225]

    )

])

# =====================================================
# Dataset
# =====================================================


class SentinelPatchDataset(Dataset):

    def __init__(

        self,

        patch_folder,

        transform=None

    ):

        self.patch_folder = Path(patch_folder)

        self.transform = transform

        self.image_paths = sorted(

            self.patch_folder.glob("*.png")

        )

    def __len__(self):

        return len(

            self.image_paths

        )

    def __getitem__(

        self,

        index

    ):

        image_path = self.image_paths[index]

        image = Image.open(

            image_path

        ).convert("RGB")

        if self.transform:

            image = self.transform(

                image

            )

        return (

            image,

            image_path.name

        )

# =====================================================
# Function
# =====================================================


def create_dataloader(folder_name):

    dataset = SentinelPatchDataset(

        PATCH_FOLDER / folder_name,

        transform=inference_transform

    )

    loader = DataLoader(

        dataset,

        batch_size=8,

        shuffle=False,

        num_workers=0,

        pin_memory=True

    )

    return loader

# =====================================================
# Load Trained Model
# =====================================================


def load_trained_model():

    print("\n" + "=" * 60)
    print("LOADING TRAINED MODEL")
    print("=" * 60)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(checkpoint)

    model.to(device)

    model.eval()

    print("Model Loaded Successfully!")

    return model


# =====================================================
# Read Metadata
# =====================================================

def load_metadata(folder_name):

    metadata_path = METADATA_FOLDER / f"{folder_name}_patches.csv"

    metadata = {}

    with open(
        metadata_path,
        "r",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            metadata[row["patch_name"]] = (

                int(row["x"]),

                int(row["y"])

            )

    return metadata


# =====================================================
# Inference Function
# =====================================================

def run_inference(folder_name):

    print("\n" + "=" * 60)
    print(f"RUNNING INFERENCE : {folder_name}")
    print("=" * 60)

    loader = create_dataloader(folder_name)

    metadata = load_metadata(folder_name)

    output_csv = OUTPUT_FOLDER / f"{folder_name}_predictions.csv"

    net = load_trained_model()

    total_images = len(loader.dataset)

    processed = 0

    with open(
        output_csv,
        "w",
        newline=""
    ) as csvfile:

        writer = csv.writer(csvfile)

        writer.writerow(

            [

                "patch_name",

                "x",

                "y",

                "predicted_class",

                "confidence"

            ]

        )

        with torch.no_grad():

            for images, patch_names in loader:

                images = images.to(device)

                outputs = net(images)

                probabilities = torch.softmax(

                    outputs,

                    dim=1

                )

                confidence, predictions = torch.max(

                    probabilities,

                    dim=1

                )

                for i in range(len(patch_names)):

                    patch = patch_names[i]

                    x, y = metadata[patch]

                    predicted_class = CLASS_NAMES[

                        predictions[i].item()

                    ]

                    writer.writerow(

                        [

                            patch,

                            x,

                            y,

                            predicted_class,

                            round(

                                confidence[i].item() * 100,

                                2

                            )

                        ]

                    )

                    processed += 1

                print(

                    f"\rProcessed : {processed}/{total_images}",

                    end=""

                )

    print("\nInference Complete!")

    print(f"Results Saved : {output_csv}")


# =====================================================
# Main Function
# =====================================================

def main():

    print("=" * 60)
    print("DEFORESTATION DETECTION")
    print("RUNNING INFERENCE ON SENTINEL PATCHES")
    print("=" * 60)

    folders = [

        "2018_rgb",

        "2024_rgb"

    ]

    for folder in folders:

        run_inference(folder)

    print("\n" + "=" * 60)
    print("ALL INFERENCE COMPLETED")
    print("=" * 60)

    print("\nPrediction files saved in:")

    print(OUTPUT_FOLDER)


# =====================================================
# Entry Point
# =====================================================

if __name__ == "__main__":

    main()
