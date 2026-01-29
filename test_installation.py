#!/usr/bin/env python3
"""Quick test script to verify the WSOL implementation."""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models import create_model
from src.utils import get_device, set_seed
from src.visualization import WSOLVisualizer


def test_installation():
    """Test if all components are working correctly."""
    print("🔍 Testing Weakly Supervised Object Localization Installation")
    print("=" * 60)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Test device detection
    device = get_device("auto")
    print(f"✅ Device detected: {device}")
    
    # Test model creation
    print("\n📦 Testing model creation...")
    models_to_test = ["gradcam", "gradcam++", "scorecam", "cam"]
    
    for model_name in models_to_test:
        try:
            model = create_model(model_name, num_classes=10)
            model.to(device)
            model.eval()
            print(f"✅ {model_name.upper()} model created successfully")
        except Exception as e:
            print(f"❌ Failed to create {model_name}: {e}")
            return False
    
    # Test forward pass
    print("\n🚀 Testing forward pass...")
    model = create_model("gradcam", num_classes=10)
    model.to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    try:
        with torch.no_grad():
            # Test classification
            logits = model(dummy_input)
            print(f"✅ Classification output shape: {logits.shape}")
            
            # Test attention map generation
            attention_map = model.get_attention_maps(dummy_input)
            print(f"✅ Attention map shape: {attention_map.shape}")
            
            # Verify attention map properties
            assert attention_map.shape == (1, 1, 224, 224), "Wrong attention map shape"
            assert torch.all(attention_map >= 0), "Attention map has negative values"
            assert torch.all(attention_map <= 1), "Attention map has values > 1"
            print("✅ Attention map properties verified")
            
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False
    
    # Test visualization
    print("\n🎨 Testing visualization...")
    try:
        visualizer = WSOLVisualizer("assets")
        print("✅ Visualizer created successfully")
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        return False
    
    # Test all models with attention maps
    print("\n🔬 Testing all WSOL methods...")
    attention_maps = {}
    
    for model_name in models_to_test:
        try:
            model = create_model(model_name, num_classes=10)
            model.to(device)
            model.eval()
            
            with torch.no_grad():
                attention_map = model.get_attention_maps(dummy_input)
                attention_maps[model_name] = attention_map.cpu().numpy()
            
            print(f"✅ {model_name.upper()} attention map generated")
            
        except Exception as e:
            print(f"❌ {model_name} failed: {e}")
            return False
    
    # Compare attention maps
    print("\n📊 Comparing attention maps...")
    for name, att_map in attention_maps.items():
        print(f"{name:12}: min={att_map.min():.3f}, max={att_map.max():.3f}, mean={att_map.mean():.3f}")
    
    print("\n🎉 All tests passed! Installation is working correctly.")
    print("\nNext steps:")
    print("1. Run training: python src/train.py")
    print("2. Run evaluation: python src/eval.py")
    print("3. Launch demo: streamlit run src/demo.py")
    print("4. Check out the notebook: jupyter notebook notebooks/example.ipynb")
    
    return True


if __name__ == "__main__":
    success = test_installation()
    sys.exit(0 if success else 1)
