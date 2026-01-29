"""Models for weakly supervised object localization."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

import timm
from torchvision import models


class BaseWSOLModel(nn.Module):
    """Base class for weakly supervised object localization models."""
    
    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            Classification logits.
        """
        raise NotImplementedError
    
    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Get attention maps for localization.
        
        Args:
            x: Input tensor.
            
        Returns:
            Attention maps.
        """
        raise NotImplementedError


class GradCAMModel(BaseWSOLModel):
    """Grad-CAM implementation for weakly supervised object localization."""
    
    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 1000,
        pretrained: bool = True,
    ) -> None:
        super().__init__(num_classes)
        
        if backbone.startswith("resnet"):
            self.backbone = getattr(models, backbone)(pretrained=pretrained)
            self.feature_layer = "layer4"
            self.target_layer = "layer4.2.conv3"
        elif backbone.startswith("vgg"):
            self.backbone = getattr(models, backbone)(pretrained=pretrained)
            self.feature_layer = "features"
            self.target_layer = "features.29"
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Replace classifier
        if hasattr(self.backbone, "fc"):
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        elif hasattr(self.backbone, "classifier"):
            self.backbone.classifier[-1] = nn.Linear(
                self.backbone.classifier[-1].in_features, num_classes
            )
        
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self) -> None:
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Get target layer
        target_layer = self._get_target_layer()
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)
    
    def _get_target_layer(self) -> nn.Module:
        """Get target layer for Grad-CAM."""
        layer_names = self.target_layer.split(".")
        layer = self.backbone
        
        for name in layer_names:
            if name.isdigit():
                layer = layer[int(name)]
            else:
                layer = getattr(layer, name)
        
        return layer
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.backbone(x)
    
    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Generate Grad-CAM attention maps.
        
        Args:
            x: Input tensor.
            
        Returns:
            Attention maps.
        """
        self.eval()
        
        # Forward pass
        logits = self.forward(x)
        
        # Get predicted class
        _, predicted_class = torch.max(logits, 1)
        
        # Backward pass
        self.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, predicted_class.unsqueeze(1), 1.0)
        
        logits.backward(gradient=one_hot, retain_graph=True)
        
        # Generate attention map
        if self.gradients is not None and self.activations is not None:
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
            attention_map = torch.sum(weights * self.activations, dim=1, keepdim=True)
            attention_map = F.relu(attention_map)
            
            # Normalize
            attention_map = F.interpolate(
                attention_map, size=x.shape[2:], mode="bilinear", align_corners=False
            )
            attention_map = normalize_image(attention_map)
            
            return attention_map
        
        return torch.zeros_like(x[:, :1])


class GradCAMPlusPlusModel(BaseWSOLModel):
    """Grad-CAM++ implementation for improved localization."""
    
    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 1000,
        pretrained: bool = True,
    ) -> None:
        super().__init__(num_classes)
        
        if backbone.startswith("resnet"):
            self.backbone = getattr(models, backbone)(pretrained=pretrained)
            self.target_layer = "layer4.2.conv3"
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Replace classifier
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self) -> None:
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Get target layer
        target_layer = self._get_target_layer()
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)
    
    def _get_target_layer(self) -> nn.Module:
        """Get target layer for Grad-CAM++."""
        layer_names = self.target_layer.split(".")
        layer = self.backbone
        
        for name in layer_names:
            if name.isdigit():
                layer = layer[int(name)]
            else:
                layer = getattr(layer, name)
        
        return layer
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.backbone(x)
    
    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Generate Grad-CAM++ attention maps.
        
        Args:
            x: Input tensor.
            
        Returns:
            Attention maps.
        """
        self.eval()
        
        # Forward pass
        logits = self.forward(x)
        
        # Get predicted class
        _, predicted_class = torch.max(logits, 1)
        
        # Backward pass
        self.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, predicted_class.unsqueeze(1), 1.0)
        
        logits.backward(gradient=one_hot, retain_graph=True)
        
        # Generate attention map using Grad-CAM++
        if self.gradients is not None and self.activations is not None:
            # Calculate alpha coefficients
            gradients_power_2 = self.gradients.pow(2)
            gradients_power_3 = self.gradients.pow(3)
            
            # Global average pooling
            sum_gradients = torch.sum(self.gradients, dim=(2, 3), keepdim=True)
            sum_gradients_power_2 = torch.sum(gradients_power_2, dim=(2, 3), keepdim=True)
            sum_gradients_power_3 = torch.sum(gradients_power_3, dim=(2, 3), keepdim=True)
            
            # Calculate alpha
            alpha = gradients_power_2 / (2 * gradients_power_2 + 
                                        sum_gradients_power_2 * self.activations)
            
            # Calculate beta
            beta = gradients_power_3 / (6 * gradients_power_3 + 
                                      sum_gradients_power_3 * self.activations)
            
            # Calculate weights
            weights = alpha * beta * torch.sum(self.gradients, dim=(2, 3), keepdim=True)
            
            # Generate attention map
            attention_map = torch.sum(weights * self.activations, dim=1, keepdim=True)
            attention_map = F.relu(attention_map)
            
            # Normalize
            attention_map = F.interpolate(
                attention_map, size=x.shape[2:], mode="bilinear", align_corners=False
            )
            attention_map = normalize_image(attention_map)
            
            return attention_map
        
        return torch.zeros_like(x[:, :1])


class ScoreCAMModel(BaseWSOLModel):
    """Score-CAM implementation for gradient-free localization."""
    
    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 1000,
        pretrained: bool = True,
    ) -> None:
        super().__init__(num_classes)
        
        if backbone.startswith("resnet"):
            self.backbone = getattr(models, backbone)(pretrained=pretrained)
            self.target_layer = "layer4.2.conv3"
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Replace classifier
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
        self.activations: Optional[torch.Tensor] = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self) -> None:
        """Register forward hooks."""
        def forward_hook(module, input, output):
            self.activations = output
        
        # Get target layer
        target_layer = self._get_target_layer()
        target_layer.register_forward_hook(forward_hook)
    
    def _get_target_layer(self) -> nn.Module:
        """Get target layer for Score-CAM."""
        layer_names = self.target_layer.split(".")
        layer = self.backbone
        
        for name in layer_names:
            if name.isdigit():
                layer = layer[int(name)]
            else:
                layer = getattr(layer, name)
        
        return layer
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.backbone(x)
    
    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Generate Score-CAM attention maps.
        
        Args:
            x: Input tensor.
            
        Returns:
            Attention maps.
        """
        self.eval()
        
        # Forward pass
        logits = self.forward(x)
        
        # Get predicted class
        _, predicted_class = torch.max(logits, 1)
        
        if self.activations is not None:
            # Generate attention map using Score-CAM
            batch_size = x.shape[0]
            attention_maps = []
            
            for i in range(batch_size):
                # Get activations for this sample
                activations = self.activations[i:i+1]
                
                # Normalize activations
                normalized_activations = F.interpolate(
                    activations, size=x.shape[2:], mode="bilinear", align_corners=False
                )
                normalized_activations = F.relu(normalized_activations)
                normalized_activations = normalize_image(normalized_activations)
                
                # Calculate scores
                scores = []
                for j in range(activations.shape[1]):
                    # Create masked input
                    mask = normalized_activations[:, j:j+1]
                    masked_input = x[i:i+1] * mask
                    
                    # Forward pass with masked input
                    with torch.no_grad():
                        masked_logits = self.forward(masked_input)
                        score = F.softmax(masked_logits, dim=1)[0, predicted_class[i]]
                        scores.append(score)
                
                scores = torch.stack(scores)
                
                # Generate attention map
                attention_map = torch.sum(
                    scores.view(-1, 1, 1, 1) * normalized_activations, dim=0, keepdim=True
                )
                attention_map = normalize_image(attention_map)
                attention_maps.append(attention_map)
            
            return torch.cat(attention_maps, dim=0)
        
        return torch.zeros_like(x[:, :1])


class CAMModel(BaseWSOLModel):
    """Class Activation Mapping (CAM) implementation."""
    
    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 1000,
        pretrained: bool = True,
    ) -> None:
        super().__init__(num_classes)
        
        if backbone.startswith("resnet"):
            self.backbone = getattr(models, backbone)(pretrained=pretrained)
            # Remove the last two layers (avgpool and fc)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
            self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(2048, num_classes)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        self.activations: Optional[torch.Tensor] = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self) -> None:
        """Register forward hooks."""
        def forward_hook(module, input, output):
            self.activations = output
        
        # Register hook on the last conv layer
        self.backbone[-1].register_forward_hook(forward_hook)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.backbone(x)
        pooled = self.global_avg_pool(features)
        flattened = pooled.view(pooled.size(0), -1)
        logits = self.classifier(flattened)
        return logits
    
    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Generate CAM attention maps.
        
        Args:
            x: Input tensor.
            
        Returns:
            Attention maps.
        """
        self.eval()
        
        # Forward pass
        logits = self.forward(x)
        
        # Get predicted class
        _, predicted_class = torch.max(logits, 1)
        
        if self.activations is not None:
            # Get weights for predicted class
            weights = self.classifier.weight[predicted_class]
            
            # Generate attention map
            attention_map = torch.sum(
                weights.view(-1, 1, 1) * self.activations, dim=1, keepdim=True
            )
            attention_map = F.relu(attention_map)
            
            # Normalize
            attention_map = F.interpolate(
                attention_map, size=x.shape[2:], mode="bilinear", align_corners=False
            )
            attention_map = normalize_image(attention_map)
            
            return attention_map
        
        return torch.zeros_like(x[:, :1])


def normalize_image(image: torch.Tensor) -> torch.Tensor:
    """Normalize image tensor to [0, 1] range."""
    return (image - image.min()) / (image.max() - image.min() + 1e-8)


def create_model(
    model_name: str,
    num_classes: int = 1000,
    pretrained: bool = True,
) -> BaseWSOLModel:
    """Create a WSOL model.
    
    Args:
        model_name: Name of the model to create.
        num_classes: Number of classes.
        pretrained: Whether to use pretrained weights.
        
    Returns:
        WSOL model instance.
    """
    model_map = {
        "gradcam": GradCAMModel,
        "gradcam++": GradCAMPlusPlusModel,
        "scorecam": ScoreCAMModel,
        "cam": CAMModel,
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model_map[model_name](
        backbone="resnet50",
        num_classes=num_classes,
        pretrained=pretrained,
    )
