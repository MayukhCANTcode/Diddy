import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

class DeforestationDataset(Dataset):
    """
    PyTorch Dataset for Sentinel-2 satellite image patches and corresponding forest masks.
    Works with GeoTIFF files using Rasterio.
    """
    def __init__(
        self,
        metadata_csv: Union[str, Path],
        img_dir: Union[str, Path],
        mask_dir: Optional[Union[str, Path]] = None,
        bands: List[str] = ["B02", "B03", "B04", "B08"],
        transform = None
    ):
        """
        Args:
            metadata_csv: Path to CSV containing patch file names and metadata (splits).
            img_dir: Directory containing processed Sentinel-2 patch files (GeoTIFFs).
            mask_dir: Directory containing ground truth forest/deforestation binary masks.
            bands: List of Sentinel-2 band names to read and stack (e.g. RGB + NIR).
            transform: Optional Albumentations transforms for augmentation.
        """
        self.metadata = pd.read_csv(metadata_csv) if Path(metadata_csv).exists() else pd.DataFrame()
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.bands = bands
        self.transform = transform

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Loads multispectral bands, stacks them, applies transforms, and returns (image_tensor, mask_tensor).
        """
        # Standard implementation will load bands using rasterio
        # Example shape: Image [C, H, W], Mask [1, H, W]
        raise NotImplementedError("Implement __getitem__ to load GeoTIFF band files.")


class EuroSATDataset(Dataset):
    """
    PyTorch Dataset for EuroSAT Land Cover Classification.
    Supports 3-band (RGB) and 13-band (Multispectral) versions.
    """
    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "train",
        multispectral: bool = True,
        transform = None
    ):
        """
        Args:
            root_dir: Path to extracted EuroSAT dataset.
            split: Split choice ('train', 'val', 'test').
            multispectral: Load full 13 bands (True) or RGB (False).
            transform: Transforms to apply.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.multispectral = multispectral
        self.transform = transform
        self.classes = []  # EuroSAT has 10 classes
        self.samples = []  # List of tuples (file_path, class_idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Loads imagery sample and returns (image_tensor, class_label).
        """
        raise NotImplementedError("Implement __getitem__ to load EuroSAT sample.")
