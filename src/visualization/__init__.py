"""Visualization utilities for weakly supervised object localization."""

import os
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class WSOLVisualizer:
    """Visualizer for weakly supervised object localization results."""
    
    def __init__(self, save_dir: str = "assets") -> None:
        """Initialize visualizer.
        
        Args:
            save_dir: Directory to save visualizations.
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def visualize_attention_maps(
        self,
        images: torch.Tensor,
        attention_maps: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        class_names: List[str],
        save_path: Optional[str] = None,
        num_samples: int = 8,
    ) -> None:
        """Visualize attention maps overlaid on images.
        
        Args:
            images: Input images.
            attention_maps: Generated attention maps.
            predictions: Model predictions.
            targets: Ground truth labels.
            class_names: List of class names.
            save_path: Path to save visualization.
            num_samples: Number of samples to visualize.
        """
        # Convert tensors to numpy
        images_np = images.cpu().numpy()
        attention_maps_np = attention_maps.cpu().numpy()
        predictions_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        # Denormalize images
        images_denorm = self._denormalize_images(images_np)
        
        # Create subplots
        fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 3, 6))
        
        for i in range(min(num_samples, len(images))):
            # Original image
            axes[0, i].imshow(images_denorm[i])
            axes[0, i].set_title(f"GT: {class_names[targets_np[i]]}")
            axes[0, i].axis("off")
            
            # Attention map overlay
            axes[1, i].imshow(images_denorm[i])
            axes[1, i].imshow(attention_maps_np[i, 0], alpha=0.6, cmap="jet")
            axes[1, i].set_title(f"Pred: {class_names[predictions_np[i]]}")
            axes[1, i].axis("off")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.savefig(f"{self.save_dir}/attention_maps.png", dpi=300, bbox_inches="tight")
        
        plt.close()
    
    def visualize_attention_heatmap(
        self,
        attention_map: np.ndarray,
        image: np.ndarray,
        title: str = "Attention Heatmap",
        save_path: Optional[str] = None,
    ) -> None:
        """Visualize attention map as heatmap.
        
        Args:
            attention_map: Attention map.
            image: Original image.
            title: Plot title.
            save_path: Path to save visualization.
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        
        # Attention map
        im = axes[1].imshow(attention_map, cmap="jet")
        axes[1].set_title("Attention Map")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1])
        
        # Overlay
        axes[2].imshow(image)
        axes[2].imshow(attention_map, alpha=0.6, cmap="jet")
        axes[2].set_title("Overlay")
        axes[2].axis("off")
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.savefig(f"{self.save_dir}/attention_heatmap.png", dpi=300, bbox_inches="tight")
        
        plt.close()
    
    def create_attention_gif(
        self,
        images: torch.Tensor,
        attention_maps: torch.Tensor,
        save_path: str,
        duration: int = 1000,
    ) -> None:
        """Create animated GIF showing attention maps.
        
        Args:
            images: Input images.
            attention_maps: Generated attention maps.
            save_path: Path to save GIF.
            duration: Frame duration in milliseconds.
        """
        images_np = images.cpu().numpy()
        attention_maps_np = attention_maps.cpu().numpy()
        
        # Denormalize images
        images_denorm = self._denormalize_images(images_np)
        
        frames = []
        for i in range(len(images)):
            # Create overlay
            overlay = images_denorm[i].copy()
            attention_map = attention_maps_np[i, 0]
            
            # Apply colormap
            attention_colored = plt.cm.jet(attention_map)[:, :, :3]
            attention_colored = (attention_colored * 255).astype(np.uint8)
            
            # Blend images
            blended = cv2.addWeighted(overlay, 0.7, attention_colored, 0.3, 0)
            frames.append(Image.fromarray(blended))
        
        # Save as GIF
        frames[0].save(
            save_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
        )
    
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        save_path: Optional[str] = None,
    ) -> None:
        """Plot training history.
        
        Args:
            history: Training history dictionary.
            save_path: Path to save plot.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss plot
        axes[0].plot(history["train_losses"], label="Train Loss")
        if "val_losses" in history:
            axes[0].plot(history["val_losses"], label="Val Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training Loss")
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy plot
        if "train_accuracies" in history:
            axes[1].plot(history["train_accuracies"], label="Train Accuracy")
        if "val_accuracies" in history:
            axes[1].plot(history["val_accuracies"], label="Val Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Training Accuracy")
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.savefig(f"{self.save_dir}/training_history.png", dpi=300, bbox_inches="tight")
        
        plt.close()
    
    def plot_metrics_comparison(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None,
    ) -> None:
        """Plot metrics comparison across models.
        
        Args:
            metrics_dict: Dictionary of model metrics.
            save_path: Path to save plot.
        """
        models = list(metrics_dict.keys())
        metrics = list(metrics_dict[models[0]].keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics[:4]):  # Plot first 4 metrics
            values = [metrics_dict[model].get(metric, 0) for model in models]
            
            bars = axes[i].bar(models, values)
            axes[i].set_title(metric.replace("_", " ").title())
            axes[i].set_ylabel("Score")
            axes[i].tick_params(axis="x", rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f"{value:.3f}", ha="center", va="bottom")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.savefig(f"{self.save_dir}/metrics_comparison.png", dpi=300, bbox_inches="tight")
        
        plt.close()
    
    def create_interactive_plot(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None,
    ) -> None:
        """Create interactive plotly visualization.
        
        Args:
            metrics_dict: Dictionary of model metrics.
            save_path: Path to save HTML file.
        """
        models = list(metrics_dict.keys())
        metrics = list(metrics_dict[models[0]].keys())
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[metric.replace("_", " ").title() for metric in metrics[:4]],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        for i, metric in enumerate(metrics[:4]):
            row = i // 2 + 1
            col = i % 2 + 1
            
            values = [metrics_dict[model].get(metric, 0) for model in models]
            
            fig.add_trace(
                go.Bar(x=models, y=values, name=metric),
                row=row, col=col
            )
        
        fig.update_layout(
            title="Model Performance Comparison",
            showlegend=False,
            height=600,
            width=800
        )
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.write_html(f"{self.save_dir}/interactive_metrics.html")
    
    def _denormalize_images(self, images: np.ndarray) -> np.ndarray:
        """Denormalize ImageNet normalized images.
        
        Args:
            images: Normalized images.
            
        Returns:
            Denormalized images.
        """
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        images_denorm = images.copy()
        for i in range(images.shape[0]):
            for j in range(3):
                images_denorm[i, j] = images_denorm[i, j] * std[j] + mean[j]
        
        # Clip to [0, 1] and convert to uint8
        images_denorm = np.clip(images_denorm, 0, 1)
        images_denorm = (images_denorm * 255).astype(np.uint8)
        
        # Transpose to HWC format
        images_denorm = np.transpose(images_denorm, (0, 2, 3, 1))
        
        return images_denorm


def create_attention_visualization(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: List[str],
    num_samples: int = 8,
    save_dir: str = "assets",
) -> None:
    """Create attention visualization for a model.
    
    Args:
        model: Model to visualize.
        dataloader: Data loader.
        device: Device to run on.
        class_names: List of class names.
        num_samples: Number of samples to visualize.
        save_dir: Directory to save visualizations.
    """
    model.eval()
    visualizer = WSOLVisualizer(save_dir)
    
    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Get predictions and attention maps
            logits = model(images)
            predictions = torch.argmax(logits, dim=1)
            
            if hasattr(model, 'get_attention_maps'):
                attention_maps = model.get_attention_maps(images)
            else:
                attention_maps = torch.zeros(images.shape[0], 1, images.shape[2], images.shape[3])
            
            # Visualize
            visualizer.visualize_attention_maps(
                images[:num_samples],
                attention_maps[:num_samples],
                predictions[:num_samples],
                labels[:num_samples],
                class_names,
            )
            
            break  # Only process first batch
