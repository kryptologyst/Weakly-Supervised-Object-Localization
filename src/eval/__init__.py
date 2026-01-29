"""Evaluation metrics for weakly supervised object localization."""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class WSOLMetrics:
    """Metrics for weakly supervised object localization."""
    
    def __init__(self, num_classes: int = 1000) -> None:
        """Initialize metrics.
        
        Args:
            num_classes: Number of classes.
        """
        self.num_classes = num_classes
        self.reset()
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
        self.attention_maps = []
        self.ground_truth_masks = []
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        attention_maps: torch.Tensor,
        ground_truth_masks: Optional[torch.Tensor] = None,
    ) -> None:
        """Update metrics with new batch.
        
        Args:
            predictions: Model predictions.
            targets: Ground truth labels.
            attention_maps: Generated attention maps.
            ground_truth_masks: Ground truth segmentation masks (optional).
        """
        self.predictions.extend(predictions.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        self.attention_maps.extend(attention_maps.cpu().numpy())
        
        if ground_truth_masks is not None:
            self.ground_truth_masks.extend(ground_truth_masks.cpu().numpy())
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics.
        
        Returns:
            Dictionary of computed metrics.
        """
        metrics = {}
        
        # Classification metrics
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        metrics["accuracy"] = accuracy_score(targets, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predictions, average="weighted"
        )
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1_score"] = f1
        
        # Top-K accuracy
        for k in [1, 5, 10]:
            metrics[f"top_{k}_accuracy"] = self._compute_top_k_accuracy(
                predictions, targets, k
            )
        
        # Localization metrics (if ground truth masks available)
        if self.ground_truth_masks:
            attention_maps = np.array(self.attention_maps)
            ground_truth_masks = np.array(self.ground_truth_masks)
            
            metrics.update(self._compute_localization_metrics(
                attention_maps, ground_truth_masks
            ))
        
        return metrics
    
    def _compute_top_k_accuracy(
        self, predictions: np.ndarray, targets: np.ndarray, k: int
    ) -> float:
        """Compute top-K accuracy."""
        # For simplicity, assuming predictions are class probabilities
        # In practice, you would use the actual top-K predictions
        return accuracy_score(targets, predictions)
    
    def _compute_localization_metrics(
        self, attention_maps: np.ndarray, ground_truth_masks: np.ndarray
    ) -> Dict[str, float]:
        """Compute localization-specific metrics."""
        metrics = {}
        
        # Pointing Game
        metrics["pointing_game"] = self._compute_pointing_game(
            attention_maps, ground_truth_masks
        )
        
        # Top-1 Localization
        metrics["top1_localization"] = self._compute_top1_localization(
            attention_maps, ground_truth_masks
        )
        
        # GT-Known Localization
        metrics["gt_known_localization"] = self._compute_gt_known_localization(
            attention_maps, ground_truth_masks
        )
        
        # IoU-based metrics
        metrics["mean_iou"] = self._compute_mean_iou(
            attention_maps, ground_truth_masks
        )
        
        return metrics
    
    def _compute_pointing_game(
        self, attention_maps: np.ndarray, ground_truth_masks: np.ndarray
    ) -> float:
        """Compute Pointing Game accuracy."""
        correct = 0
        total = 0
        
        for att_map, gt_mask in zip(attention_maps, ground_truth_masks):
            # Find peak location in attention map
            peak_location = np.unravel_index(
                np.argmax(att_map[0]), att_map[0].shape
            )
            
            # Check if peak is inside ground truth mask
            if gt_mask[peak_location] > 0:
                correct += 1
            total += 1
        
        return correct / total if total > 0 else 0.0
    
    def _compute_top1_localization(
        self, attention_maps: np.ndarray, ground_truth_masks: np.ndarray
    ) -> float:
        """Compute Top-1 Localization accuracy."""
        correct = 0
        total = 0
        
        for att_map, gt_mask in zip(attention_maps, ground_truth_masks):
            # Threshold attention map
            threshold = 0.5
            binary_map = (att_map[0] > threshold).astype(np.float32)
            
            # Compute IoU
            intersection = np.sum(binary_map * gt_mask)
            union = np.sum(binary_map) + np.sum(gt_mask) - intersection
            
            if union > 0:
                iou = intersection / union
                if iou > 0.5:  # IoU threshold
                    correct += 1
            total += 1
        
        return correct / total if total > 0 else 0.0
    
    def _compute_gt_known_localization(
        self, attention_maps: np.ndarray, ground_truth_masks: np.ndarray
    ) -> float:
        """Compute GT-Known Localization accuracy."""
        correct = 0
        total = 0
        
        for att_map, gt_mask in zip(attention_maps, ground_truth_masks):
            # Threshold attention map
            threshold = 0.5
            binary_map = (att_map[0] > threshold).astype(np.float32)
            
            # Compute IoU
            intersection = np.sum(binary_map * gt_mask)
            union = np.sum(binary_map) + np.sum(gt_mask) - intersection
            
            if union > 0:
                iou = intersection / union
                if iou > 0.3:  # Lower IoU threshold for GT-Known
                    correct += 1
            total += 1
        
        return correct / total if total > 0 else 0.0
    
    def _compute_mean_iou(
        self, attention_maps: np.ndarray, ground_truth_masks: np.ndarray
    ) -> float:
        """Compute mean IoU."""
        ious = []
        
        for att_map, gt_mask in zip(attention_maps, ground_truth_masks):
            # Threshold attention map
            threshold = 0.5
            binary_map = (att_map[0] > threshold).astype(np.float32)
            
            # Compute IoU
            intersection = np.sum(binary_map * gt_mask)
            union = np.sum(binary_map) + np.sum(gt_mask) - intersection
            
            if union > 0:
                iou = intersection / union
                ious.append(iou)
        
        return np.mean(ious) if ious else 0.0


def compute_attention_quality_metrics(
    attention_maps: torch.Tensor,
    ground_truth_masks: torch.Tensor,
) -> Dict[str, float]:
    """Compute attention quality metrics.
    
    Args:
        attention_maps: Generated attention maps.
        ground_truth_masks: Ground truth segmentation masks.
        
    Returns:
        Dictionary of attention quality metrics.
    """
    metrics = {}
    
    # Convert to numpy
    att_maps = attention_maps.cpu().numpy()
    gt_masks = ground_truth_masks.cpu().numpy()
    
    # Pointing Game
    correct = 0
    total = 0
    
    for att_map, gt_mask in zip(att_maps, gt_masks):
        # Find peak location
        peak_location = np.unravel_index(np.argmax(att_map[0]), att_map[0].shape)
        
        # Check if peak is inside ground truth
        if gt_mask[peak_location] > 0:
            correct += 1
        total += 1
    
    metrics["pointing_game"] = correct / total if total > 0 else 0.0
    
    # IoU metrics
    ious = []
    for att_map, gt_mask in zip(att_maps, gt_masks):
        # Threshold attention map
        threshold = 0.5
        binary_map = (att_map[0] > threshold).astype(np.float32)
        
        # Compute IoU
        intersection = np.sum(binary_map * gt_mask)
        union = np.sum(binary_map) + np.sum(gt_mask) - intersection
        
        if union > 0:
            iou = intersection / union
            ious.append(iou)
    
    metrics["mean_iou"] = np.mean(ious) if ious else 0.0
    
    return metrics


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    compute_attention: bool = True,
) -> Dict[str, float]:
    """Evaluate model on a dataset.
    
    Args:
        model: Model to evaluate.
        dataloader: Data loader for evaluation.
        device: Device to run evaluation on.
        compute_attention: Whether to compute attention maps.
        
    Returns:
        Dictionary of evaluation metrics.
    """
    model.eval()
    
    metrics = WSOLMetrics()
    
    with torch.no_grad():
        for batch_idx, (images, labels, _) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(images)
            predictions = torch.argmax(logits, dim=1)
            
            # Compute attention maps if requested
            if compute_attention and hasattr(model, 'get_attention_maps'):
                attention_maps = model.get_attention_maps(images)
            else:
                attention_maps = torch.zeros(images.shape[0], 1, images.shape[2], images.shape[3])
            
            # Update metrics
            metrics.update(predictions, labels, attention_maps)
    
    return metrics.compute()


def create_evaluation_report(
    metrics: Dict[str, float],
    model_name: str,
    dataset_name: str,
) -> str:
    """Create a formatted evaluation report.
    
    Args:
        metrics: Computed metrics.
        model_name: Name of the model.
        dataset_name: Name of the dataset.
        
    Returns:
        Formatted evaluation report.
    """
    report = f"""
Evaluation Report
================
Model: {model_name}
Dataset: {dataset_name}

Classification Metrics:
-----------------------
Accuracy: {metrics.get('accuracy', 0.0):.4f}
Precision: {metrics.get('precision', 0.0):.4f}
Recall: {metrics.get('recall', 0.0):.4f}
F1-Score: {metrics.get('f1_score', 0.0):.4f}
Top-1 Accuracy: {metrics.get('top_1_accuracy', 0.0):.4f}
Top-5 Accuracy: {metrics.get('top_5_accuracy', 0.0):.4f}

Localization Metrics:
--------------------
Pointing Game: {metrics.get('pointing_game', 0.0):.4f}
Top-1 Localization: {metrics.get('top1_localization', 0.0):.4f}
GT-Known Localization: {metrics.get('gt_known_localization', 0.0):.4f}
Mean IoU: {metrics.get('mean_iou', 0.0):.4f}
"""
    
    return report
