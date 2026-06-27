import numpy as np
import torch
from typing import Dict, Union

class SegmentationMetrics:
    """
    Computes and aggregates standard semantic segmentation metrics including
    Intersection over Union (IoU), Precision, Recall, and F1-score.
    """
    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.total_tp = np.zeros(self.num_classes)
        self.total_fp = np.zeros(self.num_classes)
        self.total_fn = np.zeros(self.num_classes)
        self.total_tn = np.zeros(self.num_classes)

    def update(self, preds: Union[torch.Tensor, np.ndarray], targets: Union[torch.Tensor, np.ndarray]):
        """
        Updates running confusion parameters.
        
        Args:
            preds: Predicted class labels (binary/multiclass map).
            targets: True ground truth class labels.
        """
        # Convert tensors to numpy if needed
        # Compute true positive, false positive, true negative, false negative for each class
        pass

    def compute(self) -> Dict[str, np.ndarray]:
        """
        Computes precision, recall, F1, and IoU scores across all accumulated predictions.
        
        Returns:
            Dictionary containing metrics arrays per class.
        """
        # Compute metrics
        # IoU = TP / (TP + FP + FN)
        # Precision = TP / (TP + FP)
        # Recall = TP / (TP + FN)
        # F1 = 2 * (Precision * Recall) / (Precision + Recall)
        return {
            "iou": np.zeros(self.num_classes),
            "precision": np.zeros(self.num_classes),
            "recall": np.zeros(self.num_classes),
            "f1": np.zeros(self.num_classes)
        }


def calculate_confusion_matrix(preds: np.ndarray, targets: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Computes a standard confusion matrix.
    
    Args:
        preds: Flattened predicted labels.
        targets: Flattened ground truth labels.
        num_classes: Number of distinct classes.
        
    Returns:
        Confusion matrix of shape (num_classes, num_classes) where rows are targets and columns are predictions.
    """
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    # Compute confusion values
    return matrix
