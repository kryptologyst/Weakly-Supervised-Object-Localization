"""Test suite for weakly supervised object localization."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from src.models import create_model, GradCAMModel, GradCAMPlusPlusModel, ScoreCAMModel, CAMModel
from src.data import WSOLDataset, get_transforms
from src.eval import WSOLMetrics, evaluate_model
from src.utils import set_seed, get_device, count_parameters


class TestModels:
    """Test WSOL model implementations."""
    
    def test_create_model(self):
        """Test model creation."""
        model = create_model("gradcam", num_classes=10)
        assert isinstance(model, GradCAMModel)
        assert model.num_classes == 10
    
    def test_gradcam_forward(self):
        """Test Grad-CAM forward pass."""
        model = GradCAMModel(num_classes=10)
        x = torch.randn(2, 3, 224, 224)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (2, 10)
    
    def test_gradcam_attention_maps(self):
        """Test Grad-CAM attention map generation."""
        model = GradCAMModel(num_classes=10)
        x = torch.randn(1, 3, 224, 224)
        
        attention_map = model.get_attention_maps(x)
        
        assert attention_map.shape == (1, 1, 224, 224)
        assert torch.all(attention_map >= 0)
        assert torch.all(attention_map <= 1)
    
    def test_gradcam_plus_plus(self):
        """Test Grad-CAM++ implementation."""
        model = GradCAMPlusPlusModel(num_classes=10)
        x = torch.randn(1, 3, 224, 224)
        
        attention_map = model.get_attention_maps(x)
        
        assert attention_map.shape == (1, 1, 224, 224)
        assert torch.all(attention_map >= 0)
        assert torch.all(attention_map <= 1)
    
    def test_scorecam(self):
        """Test Score-CAM implementation."""
        model = ScoreCAMModel(num_classes=10)
        x = torch.randn(1, 3, 224, 224)
        
        attention_map = model.get_attention_maps(x)
        
        assert attention_map.shape == (1, 1, 224, 224)
        assert torch.all(attention_map >= 0)
        assert torch.all(attention_map <= 1)
    
    def test_cam(self):
        """Test CAM implementation."""
        model = CAMModel(num_classes=10)
        x = torch.randn(1, 3, 224, 224)
        
        attention_map = model.get_attention_maps(x)
        
        assert attention_map.shape == (1, 1, 224, 224)
        assert torch.all(attention_map >= 0)
        assert torch.all(attention_map <= 1)


class TestData:
    """Test data loading and preprocessing."""
    
    def test_dataset_creation(self):
        """Test dataset creation."""
        dataset = WSOLDataset("data", split="train")
        assert len(dataset) > 0
    
    def test_dataset_getitem(self):
        """Test dataset item retrieval."""
        dataset = WSOLDataset("data", split="train")
        image, label, path = dataset[0]
        
        assert isinstance(image, torch.Tensor)
        assert isinstance(label, int)
        assert isinstance(path, str)
        assert image.shape == (3, 224, 224)
    
    def test_transforms(self):
        """Test image transforms."""
        transform = get_transforms("train", image_size=224)
        
        # Create dummy image
        image = torch.randn(3, 256, 256)
        
        # Apply transform
        transformed = transform(image)
        
        assert transformed.shape == (3, 224, 224)
        assert torch.all(transformed >= -3)  # Normalized range
        assert torch.all(transformed <= 3)


class TestEvaluation:
    """Test evaluation metrics."""
    
    def test_wsol_metrics(self):
        """Test WSOL metrics calculation."""
        metrics = WSOLMetrics(num_classes=10)
        
        # Mock data
        predictions = torch.tensor([0, 1, 2, 0, 1])
        targets = torch.tensor([0, 1, 2, 1, 1])
        attention_maps = torch.randn(5, 1, 224, 224)
        
        metrics.update(predictions, targets, attention_maps)
        results = metrics.compute()
        
        assert "accuracy" in results
        assert "precision" in results
        assert "recall" in results
        assert "f1_score" in results
        assert 0 <= results["accuracy"] <= 1
    
    def test_attention_quality_metrics(self):
        """Test attention quality metrics."""
        from src.eval import compute_attention_quality_metrics
        
        attention_maps = torch.randn(2, 1, 224, 224)
        ground_truth_masks = torch.randint(0, 2, (2, 224, 224)).float()
        
        metrics = compute_attention_quality_metrics(attention_maps, ground_truth_masks)
        
        assert "pointing_game" in metrics
        assert "mean_iou" in metrics
        assert 0 <= metrics["pointing_game"] <= 1
        assert 0 <= metrics["mean_iou"] <= 1


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test random seed setting."""
        set_seed(42)
        
        # Generate random numbers
        torch_rand1 = torch.randn(10)
        np_rand1 = np.random.randn(10)
        
        # Reset seed and generate again
        set_seed(42)
        torch_rand2 = torch.randn(10)
        np_rand2 = np.random.randn(10)
        
        # Should be identical
        assert torch.allclose(torch_rand1, torch_rand2)
        assert np.allclose(np_rand1, np_rand2)
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device("auto")
        assert isinstance(device, torch.device)
        
        device = get_device("cpu")
        assert device.type == "cpu"
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = GradCAMModel(num_classes=10)
        num_params = count_parameters(model)
        
        assert num_params > 0
        assert isinstance(num_params, int)


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_training(self):
        """Test end-to-end training pipeline."""
        # Create model
        model = create_model("gradcam", num_classes=10)
        
        # Create dummy data
        dataset = WSOLDataset("data", split="train")
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
        
        # Test forward pass
        for images, labels, _ in dataloader:
            logits = model(images)
            attention_maps = model.get_attention_maps(images)
            
            assert logits.shape[0] == images.shape[0]
            assert attention_maps.shape[0] == images.shape[0]
            break
    
    def test_model_evaluation(self):
        """Test model evaluation."""
        model = create_model("gradcam", num_classes=10)
        dataset = WSOLDataset("data", split="test")
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
        
        device = get_device("cpu")
        model.to(device)
        
        metrics = evaluate_model(model, dataloader, device)
        
        assert "accuracy" in metrics
        assert isinstance(metrics["accuracy"], float)


if __name__ == "__main__":
    pytest.main([__file__])
