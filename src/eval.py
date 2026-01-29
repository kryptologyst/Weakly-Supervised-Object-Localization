"""Main evaluation script for weakly supervised object localization."""

import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader

from src.models import create_model
from src.data import create_dataloaders
from src.eval import evaluate_model, create_evaluation_report
from src.utils import set_seed, get_device, create_directory_structure, load_checkpoint
from src.visualization import create_attention_visualization


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    """Main evaluation function."""
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
    
    # Load checkpoint if specified
    if config.get("checkpoint_path"):
        print(f"Loading checkpoint: {config.checkpoint_path}")
        checkpoint = load_checkpoint(config.checkpoint_path, model, device=device)
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"Best score: {checkpoint['best_score']:.4f}")
    
    print(f"Evaluating model: {config.model.name}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    # Evaluate on different splits
    splits = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }
    
    all_metrics = {}
    
    for split_name, dataloader in splits.items():
        print(f"\nEvaluating on {split_name} set...")
        
        # Evaluate model
        metrics = evaluate_model(model, dataloader, device)
        all_metrics[split_name] = metrics
        
        # Print metrics
        print(f"{split_name.capitalize()} Results:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        
        # Create attention visualizations
        if split_name == "test":
            print("Creating attention visualizations...")
            create_attention_visualization(
                model=model,
                dataloader=dataloader,
                device=device,
                class_names=[f"class_{i}" for i in range(config.data.num_classes)],
                num_samples=8,
                save_dir=config.paths.assets_dir,
            )
    
    # Create comprehensive evaluation report
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    for split_name, metrics in all_metrics.items():
        print(f"\n{split_name.upper()} SET:")
        print("-" * 20)
        
        # Classification metrics
        print("Classification Metrics:")
        print(f"  Accuracy: {metrics.get('accuracy', 0.0):.4f}")
        print(f"  Precision: {metrics.get('precision', 0.0):.4f}")
        print(f"  Recall: {metrics.get('recall', 0.0):.4f}")
        print(f"  F1-Score: {metrics.get('f1_score', 0.0):.4f}")
        
        # Localization metrics (if available)
        if 'pointing_game' in metrics:
            print("Localization Metrics:")
            print(f"  Pointing Game: {metrics['pointing_game']:.4f}")
            print(f"  Top-1 Localization: {metrics['top1_localization']:.4f}")
            print(f"  GT-Known Localization: {metrics['gt_known_localization']:.4f}")
            print(f"  Mean IoU: {metrics['mean_iou']:.4f}")
    
    # Save detailed report
    report_path = f"{config.paths.assets_dir}/evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write(create_evaluation_report(
            all_metrics["test"],
            config.model.name,
            "synthetic_dataset"
        ))
    
    print(f"\nDetailed report saved to: {report_path}")
    print(f"Visualizations saved to: {config.paths.assets_dir}")


if __name__ == "__main__":
    main()
