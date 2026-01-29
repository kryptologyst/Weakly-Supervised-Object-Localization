"""Utility functions for weakly supervised object localization."""

import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: str = "auto") -> torch.device:
    """Get the appropriate device for computation.
    
    Args:
        device: Device specification. If "auto", automatically select best available.
        
    Returns:
        PyTorch device object.
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model: nn.Module) -> str:
    """Get human-readable model size.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Model size as string (e.g., "1.2M", "25.6M").
    """
    num_params = count_parameters(model)
    
    if num_params >= 1e9:
        return f"{num_params / 1e9:.1f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.1f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.1f}K"
    else:
        return str(num_params)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file.
        model: Model to load weights into.
        optimizer: Optimizer to load state into.
        device: Device to load checkpoint on.
        
    Returns:
        Dictionary containing checkpoint information.
    """
    if device is None:
        device = next(model.parameters()).device
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    result = {
        "epoch": checkpoint.get("epoch", 0),
        "best_score": checkpoint.get("best_score", 0.0),
        "model_state_dict": checkpoint["model_state_dict"],
    }
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        result["optimizer_state_dict"] = checkpoint["optimizer_state_dict"]
    
    return result


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_score: float,
    checkpoint_path: str,
    additional_info: Optional[Dict[str, Any]] = None,
) -> None:
    """Save model checkpoint.
    
    Args:
        model: Model to save.
        optimizer: Optimizer to save.
        epoch: Current epoch.
        best_score: Best score achieved.
        checkpoint_path: Path to save checkpoint.
        additional_info: Additional information to save.
    """
    checkpoint = {
        "epoch": epoch,
        "best_score": best_score,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    
    if additional_info:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, checkpoint_path)


def create_directory_structure(config: DictConfig) -> None:
    """Create necessary directories based on configuration.
    
    Args:
        config: Configuration object.
    """
    import os
    
    directories = [
        config.paths.data_dir,
        config.paths.raw_data_dir,
        config.paths.processed_data_dir,
        config.paths.checkpoints_dir,
        config.paths.logs_dir,
        config.paths.assets_dir,
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def normalize_image(image: torch.Tensor) -> torch.Tensor:
    """Normalize image tensor to [0, 1] range.
    
    Args:
        image: Input image tensor.
        
    Returns:
        Normalized image tensor.
    """
    return (image - image.min()) / (image.max() - image.min() + 1e-8)


def denormalize_image(
    image: torch.Tensor,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """Denormalize image tensor from ImageNet normalization.
    
    Args:
        image: Normalized image tensor.
        mean: Mean values used for normalization.
        std: Standard deviation values used for normalization.
        
    Returns:
        Denormalized image tensor.
    """
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    
    if image.device != mean.device:
        mean = mean.to(image.device)
        std = std.to(image.device)
    
    return image * std + mean


def calculate_flops(model: nn.Module, input_shape: Tuple[int, ...]) -> int:
    """Calculate FLOPs for a model.
    
    Args:
        model: PyTorch model.
        input_shape: Input tensor shape (excluding batch dimension).
        
    Returns:
        Number of FLOPs.
    """
    from thop import profile
    
    dummy_input = torch.randn(1, *input_shape)
    flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
    return flops


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self) -> None:
        self.reset()
    
    def reset(self) -> None:
        """Reset all values."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
    
    def update(self, val: float, n: int = 1) -> None:
        """Update with new value.
        
        Args:
            val: New value.
            n: Number of samples.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def format_time(seconds: float) -> str:
    """Format time in human-readable format.
    
    Args:
        seconds: Time in seconds.
        
    Returns:
        Formatted time string.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"
