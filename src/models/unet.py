import torch
import torch.nn as nn
from typing import Dict, Any

class DeforestationUNet(nn.Module):
    """
    U-Net Convolutional Network for semantic segmentation of deforestation / forest cover.
    Modified to accept multispectral satellite images (e.g. 4 bands: RGB + NIR).
    """
    def __init__(self, in_channels: int = 4, num_classes: int = 2):
        """
        Args:
            in_channels: Number of input channels (typically 4 for Sentinel-2 RGB + NIR, or 13 for full Sentinel-2 bands).
            num_classes: Number of target land cover classes (e.g. 2 for intact forest vs. deforested).
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Structure will consist of contracting path (encoder), bottleneck, and expanding path (decoder)
        # Skip connections connect matching resolutions between encoder and decoder
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tensor of shape (BatchSize, in_channels, Height, Width)
            
        Returns:
            Logits tensor of shape (BatchSize, num_classes, Height, Width)
        """
        raise NotImplementedError("Implement U-Net encoder, decoder, and skip connections.")
