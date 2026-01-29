"""Main training script for weakly supervised object localization."""

import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader

from src.models import create_model
from src.data import create_dataloaders
from src.train import train_model
from src.utils import set_seed, get_device, create_directory_structure
from src.visualization import WSOLVisualizer


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    """Main training function."""
    # Set random seed for reproducibility
    set_seed(config.seed)
    
    # Create directory structure
    create_directory_structure(config)
    
    # Get device
    device = get_device(config.device)
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(
        model_name=config.model.name,
        num_classes=config.data.num_classes,
        pretrained=config.model.pretrained,
    )
    model.to(device)
    
    print(f"Created model: {config.model.name}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Train model
    print("Starting training...")
    results = train_model(model, config, device, train_loader, val_loader)
    
    # Print results
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {results['best_score']:.4f}")
    print(f"Training time: {results['history']['training_time']:.2f} seconds")
    
    # Create visualizations
    visualizer = WSOLVisualizer(config.paths.assets_dir)
    visualizer.plot_training_history(results["history"])
    
    print(f"Visualizations saved to: {config.paths.assets_dir}")


if __name__ == "__main__":
    main()
