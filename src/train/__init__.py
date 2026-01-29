"""Training utilities for weakly supervised object localization."""

import time
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from tqdm import tqdm

from .utils import AverageMeter, format_time, save_checkpoint, load_checkpoint
from .eval import evaluate_model, WSOLMetrics


class WSOLTrainer:
    """Trainer for weakly supervised object localization models."""
    
    def __init__(
        self,
        model: nn.Module,
        config: DictConfig,
        device: torch.device,
    ) -> None:
        """Initialize trainer.
        
        Args:
            model: Model to train.
            config: Training configuration.
            device: Device to train on.
        """
        self.model = model
        self.config = config
        self.device = device
        
        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.criterion = nn.CrossEntropyLoss()
        
        # Training state
        self.current_epoch = 0
        self.best_score = 0.0
        self.train_losses = []
        self.val_scores = []
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer."""
        if self.config.training.optimizer.name == "adam":
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.training.optimizer.lr,
                weight_decay=self.config.training.optimizer.weight_decay,
            )
        elif self.config.training.optimizer.name == "sgd":
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.training.optimizer.lr,
                momentum=self.config.training.optimizer.momentum,
                weight_decay=self.config.training.optimizer.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer.name}")
        
        return optimizer
    
    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler."""
        if self.config.training.scheduler.name == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.epochs,
            )
        elif self.config.training.scheduler.name == "step":
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.training.scheduler.step_size,
                gamma=self.config.training.scheduler.gamma,
            )
        elif self.config.training.scheduler.name == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=self.config.training.scheduler.factor,
                patience=self.config.training.scheduler.patience,
            )
        else:
            scheduler = None
        
        return scheduler
    
    def train_epoch(
        self,
        train_loader: DataLoader,
    ) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader.
            
        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        
        losses = AverageMeter()
        accuracies = AverageMeter()
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (images, labels, _) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            if self.scaler:
                with torch.cuda.amp.autocast():
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            # Update metrics
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == labels).float().mean().item()
            
            losses.update(loss.item(), images.size(0))
            accuracies.update(accuracy, images.size(0))
            
            # Update progress bar
            pbar.set_postfix({
                "Loss": f"{losses.avg:.4f}",
                "Acc": f"{accuracies.avg:.4f}",
            })
        
        return {
            "train_loss": losses.avg,
            "train_accuracy": accuracies.avg,
        }
    
    def validate(
        self,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        """Validate the model.
        
        Args:
            val_loader: Validation data loader.
            
        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        
        losses = AverageMeter()
        accuracies = AverageMeter()
        
        with torch.no_grad():
            for images, labels, _ in tqdm(val_loader, desc="Validation"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        logits = self.model(images)
                        loss = self.criterion(logits, labels)
                else:
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                
                # Update metrics
                predictions = torch.argmax(logits, dim=1)
                accuracy = (predictions == labels).float().mean().item()
                
                losses.update(loss.item(), images.size(0))
                accuracies.update(accuracy, images.size(0))
        
        return {
            "val_loss": losses.avg,
            "val_accuracy": accuracies.avg,
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            
        Returns:
            Training history.
        """
        start_time = time.time()
        
        for epoch in range(self.config.training.epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            self.train_losses.append(train_metrics["train_loss"])
            
            # Validate
            val_metrics = self.validate(val_loader)
            self.val_scores.append(val_metrics["val_accuracy"])
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["val_accuracy"])
                else:
                    self.scheduler.step()
            
            # Save checkpoint if best
            if val_metrics["val_accuracy"] > self.best_score:
                self.best_score = val_metrics["val_accuracy"]
                self._save_checkpoint(is_best=True)
            
            # Log metrics
            self._log_metrics(epoch, train_metrics, val_metrics)
            
            # Save regular checkpoint
            if epoch % self.config.logging.save_every_n_epochs == 0:
                self._save_checkpoint(is_best=False)
        
        training_time = time.time() - start_time
        
        return {
            "train_losses": self.train_losses,
            "val_scores": self.val_scores,
            "best_score": self.best_score,
            "training_time": training_time,
        }
    
    def _save_checkpoint(self, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint_name = "best_model.pth" if is_best else f"checkpoint_epoch_{self.current_epoch}.pth"
        checkpoint_path = f"{self.config.paths.checkpoints_dir}/{checkpoint_name}"
        
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=self.current_epoch,
            best_score=self.best_score,
            checkpoint_path=checkpoint_path,
            additional_info={
                "config": self.config,
                "train_losses": self.train_losses,
                "val_scores": self.val_scores,
            },
        )
    
    def _log_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
    ) -> None:
        """Log training metrics."""
        print(f"\nEpoch {epoch + 1}/{self.config.training.epochs}")
        print(f"Train Loss: {train_metrics['train_loss']:.4f}, Train Acc: {train_metrics['train_accuracy']:.4f}")
        print(f"Val Loss: {val_metrics['val_loss']:.4f}, Val Acc: {val_metrics['val_accuracy']:.4f}")
        print(f"Best Score: {self.best_score:.4f}")
        
        # Log to tensorboard if enabled
        if self.config.logging.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=self.config.paths.logs_dir)
            
            writer.add_scalar("Train/Loss", train_metrics["train_loss"], epoch)
            writer.add_scalar("Train/Accuracy", train_metrics["train_accuracy"], epoch)
            writer.add_scalar("Val/Loss", val_metrics["val_loss"], epoch)
            writer.add_scalar("Val/Accuracy", val_metrics["val_accuracy"], epoch)
            
            writer.close()
        
        # Log to wandb if enabled
        if self.config.logging.use_wandb:
            import wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": train_metrics["train_loss"],
                "train_accuracy": train_metrics["train_accuracy"],
                "val_loss": val_metrics["val_loss"],
                "val_accuracy": val_metrics["val_accuracy"],
                "best_score": self.best_score,
            })


def train_model(
    model: nn.Module,
    config: DictConfig,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> Dict[str, Any]:
    """Train a WSOL model.
    
    Args:
        model: Model to train.
        config: Training configuration.
        device: Device to train on.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        
    Returns:
        Training results.
    """
    trainer = WSOLTrainer(model, config, device)
    history = trainer.train(train_loader, val_loader)
    
    return {
        "model": trainer.model,
        "history": history,
        "best_score": trainer.best_score,
    }


def fine_tune_model(
    model: nn.Module,
    config: DictConfig,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    freeze_backbone: bool = True,
) -> Dict[str, Any]:
    """Fine-tune a pretrained model.
    
    Args:
        model: Model to fine-tune.
        config: Training configuration.
        device: Device to train on.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        freeze_backbone: Whether to freeze backbone parameters.
        
    Returns:
        Fine-tuning results.
    """
    if freeze_backbone:
        # Freeze backbone parameters
        for name, param in model.named_parameters():
            if "backbone" in name:
                param.requires_grad = False
    
    # Train only unfrozen parameters
    results = train_model(model, config, device, train_loader, val_loader)
    
    return results
