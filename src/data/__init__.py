"""Data loading and preprocessing for weakly supervised object localization."""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import albumentations as A
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from omegaconf import DictConfig


class WSOLDataset(Dataset):
    """Dataset for weakly supervised object localization."""
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
    ) -> None:
        """Initialize dataset.
        
        Args:
            data_dir: Directory containing the dataset.
            split: Dataset split (train, val, test).
            transform: Transform to apply to images.
            target_transform: Transform to apply to targets.
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        # Load annotations
        self.annotations = self._load_annotations()
        
        # ImageNet class names
        self.class_names = self._load_class_names()
    
    def _load_annotations(self) -> pd.DataFrame:
        """Load dataset annotations."""
        # For demo purposes, create synthetic annotations
        # In practice, you would load from CSV/JSON files
        
        if not os.path.exists(os.path.join(self.data_dir, "annotations.csv")):
            return self._create_synthetic_annotations()
        
        return pd.read_csv(os.path.join(self.data_dir, "annotations.csv"))
    
    def _create_synthetic_annotations(self) -> pd.DataFrame:
        """Create synthetic annotations for demo purposes."""
        # Create a small synthetic dataset
        annotations = []
        
        # Sample ImageNet classes
        sample_classes = [
            "n01440764", "n01443537", "n01484850", "n01491361", "n01494475",
            "n01496331", "n01498041", "n01514668", "n01514859", "n01518878"
        ]
        
        for i, class_id in enumerate(sample_classes):
            for j in range(10):  # 10 samples per class
                annotations.append({
                    "image_path": f"sample_{i}_{j}.jpg",
                    "class_id": i,
                    "class_name": class_id,
                    "split": "train" if j < 7 else "val"
                })
        
        return pd.DataFrame(annotations)
    
    def _load_class_names(self) -> List[str]:
        """Load ImageNet class names."""
        # Sample ImageNet class names for demo
        return [
            "tench", "goldfish", "great white shark", "tiger shark", "hammerhead",
            "electric ray", "stingray", "cock", "hen", "ostrich"
        ]
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.annotations[self.annotations["split"] == self.split])
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """Get dataset item.
        
        Args:
            idx: Item index.
            
        Returns:
            Tuple of (image, class_id, image_path).
        """
        split_data = self.annotations[self.annotations["split"] == self.split]
        row = split_data.iloc[idx]
        
        # Load image
        image_path = os.path.join(self.data_dir, "images", row["image_path"])
        
        # Create synthetic image if it doesn't exist
        if not os.path.exists(image_path):
            image = self._create_synthetic_image(row["class_id"])
        else:
            image = Image.open(image_path).convert("RGB")
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        class_id = row["class_id"]
        if self.target_transform:
            class_id = self.target_transform(class_id)
        
        return image, class_id, row["image_path"]
    
    def _create_synthetic_image(self, class_id: int) -> Image.Image:
        """Create synthetic image for demo purposes."""
        # Create a simple synthetic image
        np.random.seed(class_id)
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Add some structure based on class
        center_x, center_y = 112, 112
        radius = 50
        
        # Create a circle
        y, x = np.ogrid[:224, :224]
        mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
        
        # Color based on class
        color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)][class_id % 5]
        image[mask] = color
        
        return Image.fromarray(image)


def get_transforms(
    split: str = "train",
    image_size: int = 224,
    augmentation_strength: str = "medium",
) -> transforms.Compose:
    """Get image transforms for the specified split.
    
    Args:
        split: Dataset split (train, val, test).
        image_size: Target image size.
        augmentation_strength: Strength of augmentation (light, medium, strong).
        
    Returns:
        Transform composition.
    """
    if split == "train":
        if augmentation_strength == "light":
            transform_list = [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ]
        elif augmentation_strength == "medium":
            transform_list = [
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ]
        else:  # strong
            transform_list = [
                transforms.Resize((image_size + 64, image_size + 64)),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ]
    else:  # val or test
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ]
    
    return transforms.Compose(transform_list)


def get_albumentations_transforms(
    split: str = "train",
    image_size: int = 224,
    augmentation_strength: str = "medium",
) -> A.Compose:
    """Get Albumentations transforms for the specified split.
    
    Args:
        split: Dataset split (train, val, test).
        image_size: Target image size.
        augmentation_strength: Strength of augmentation (light, medium, strong).
        
    Returns:
        Albumentations transform composition.
    """
    if split == "train":
        if augmentation_strength == "light":
            transform_list = [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ]
        elif augmentation_strength == "medium":
            transform_list = [
                A.Resize(image_size + 32, image_size + 32),
                A.RandomCrop(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ]
        else:  # strong
            transform_list = [
                A.Resize(image_size + 64, image_size + 64),
                A.RandomCrop(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.7),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.5),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ]
    else:  # val or test
        transform_list = [
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ]
    
    return A.Compose(transform_list)


def create_dataloaders(
    config: DictConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for train, validation, and test sets.
    
    Args:
        config: Configuration object.
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    # Get transforms
    train_transform = get_transforms(
        split="train",
        image_size=config.data.image_size,
        augmentation_strength=config.data.augmentation_strength,
    )
    
    val_transform = get_transforms(
        split="val",
        image_size=config.data.image_size,
    )
    
    test_transform = get_transforms(
        split="test",
        image_size=config.data.image_size,
    )
    
    # Create datasets
    train_dataset = WSOLDataset(
        data_dir=config.paths.data_dir,
        split="train",
        transform=train_transform,
    )
    
    val_dataset = WSOLDataset(
        data_dir=config.paths.data_dir,
        split="val",
        transform=val_transform,
    )
    
    test_dataset = WSOLDataset(
        data_dir=config.paths.data_dir,
        split="test",
        transform=test_transform,
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
    
    return train_loader, val_loader, test_loader


def collate_fn(batch: List[Tuple[torch.Tensor, int, str]]) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Custom collate function for data loader.
    
    Args:
        batch: Batch of data.
        
    Returns:
        Tuple of (images, labels, image_paths).
    """
    images, labels, image_paths = zip(*batch)
    
    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return images, labels, list(image_paths)
