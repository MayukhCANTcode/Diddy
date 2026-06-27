import torch
import torch.nn as nn
from typing import Tuple

class BiTemporalChangeDetector(nn.Module):
    """
    Bi-Temporal change detection network.
    Compares two satellite images acquired over the same coordinates at different times (Time 1, Time 2)
    to predict deforestation / canopy loss patterns.
    """
    def __init__(self, in_channels: int = 4):
        """
        Args:
            in_channels: Number of spectral bands in each image.
        """
        super().__init__()
        self.in_channels = in_channels
        
        # Typically uses either:
        # 1. Siamese encoder style (two streams with shared weights) followed by feature difference and decoder
        # 2. Concat channel style (early fusion) passing an 8-channel image into a standard U-Net
        
    def forward(self, img_t1: torch.Tensor, img_t2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            img_t1: Satellite image at Time 1, shape: (BatchSize, in_channels, Height, Width)
            img_t2: Satellite image at Time 2, shape: (BatchSize, in_channels, Height, Width)
            
        Returns:
            Logits of change map, shape: (BatchSize, 1, Height, Width) where higher values indicate change.
        """
        raise NotImplementedError("Implement bi-temporal feature aggregation or early fusion logic.")
