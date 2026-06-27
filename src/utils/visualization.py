import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Union, Optional

def plot_prediction_vs_target(
    image: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    save_path: Optional[Union[str, Path]] = None
):
    """
    Plots a three-column panel comparing:
    1. Input Composite (e.g. Sentinel-2 RGB or False Color Infrared)
    2. Ground Truth Forest/Deforestation Mask
    3. Model Prediction Mask
    
    Args:
        image: Numpy array of shape (H, W, 3) representing RGB bands or false-color NIR.
        target: Binary numpy array of shape (H, W) for ground truth.
        prediction: Binary numpy array of shape (H, W) for predictions.
        save_path: Path to save the generated plot. If None, runs plt.show().
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # Panel 1: Composite
    # Panel 2: Ground Truth
    # Panel 3: Prediction
    # Save or show
    raise NotImplementedError("Implement matplotlib visualization canvas.")


def save_map_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    save_path: Union[str, Path],
    color: str = "red",
    alpha: float = 0.4
):
    """
    Creates and saves a semi-transparent colored overlay of the deforestation mask onto the RGB background.
    Useful for creating qualitative results for presentations and paper figures.
    
    Args:
        image: Input RGB background, shape: (H, W, 3)
        mask: Binary mask of detected deforestation, shape: (H, W)
        save_path: Location where output image is saved.
        color: Color of overlay ('red', 'yellow', etc.)
        alpha: Opacity factor (0.0 = fully transparent, 1.0 = fully opaque)
    """
    raise NotImplementedError("Implement image masking and alpha blending logic.")
