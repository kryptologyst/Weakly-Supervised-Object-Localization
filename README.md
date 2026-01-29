# Weakly Supervised Object Localization

A production-ready implementation of weakly supervised object localization (WSOL) methods including Grad-CAM, Grad-CAM++, Score-CAM, and CAM.

## Overview

Weakly supervised object localization involves identifying the locations of objects within images using only class labels or image-level annotations, without requiring full bounding box annotations. This project provides a comprehensive implementation of state-of-the-art WSOL methods with proper evaluation metrics and visualization tools.

## Features

- **Multiple WSOL Methods**: Grad-CAM, Grad-CAM++, Score-CAM, and CAM implementations
- **Modern Architecture**: Built with PyTorch 2.x, Python 3.10+, and modern best practices
- **Comprehensive Evaluation**: Classification and localization metrics including Pointing Game, Top-1 Localization, and IoU-based metrics
- **Interactive Demo**: Streamlit-based web interface for real-time visualization
- **Production Ready**: Proper configuration management, logging, and reproducible experiments
- **Extensible**: Easy to add new WSOL methods and evaluation metrics

## Installation

### Prerequisites

- Python 3.10 or higher
- PyTorch 2.0 or higher
- CUDA-capable GPU (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Weakly-Supervised-Object-Localization.git
cd Weakly-Supervised-Object-Localization
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Quick Start

### 1. Training a Model

Train a WSOL model using Hydra configuration:

```bash
python src/train.py
```

Customize training with different models:
```bash
python src/train.py model=gradcam++ training.epochs=100
```

### 2. Evaluation

Evaluate a trained model:

```bash
python src/eval.py checkpoint_path=checkpoints/best_model.pth
```

### 3. Interactive Demo

Launch the Streamlit demo:

```bash
streamlit run src/demo.py
```

## Project Structure

```
├── src/                    # Source code
│   ├── models/            # WSOL model implementations
│   ├── data/              # Data loading and preprocessing
│   ├── train/             # Training utilities
│   ├── eval/              # Evaluation metrics and tools
│   ├── visualization/     # Visualization utilities
│   ├── demo/              # Demo interface
│   └── utils/             # Utility functions
├── configs/               # Configuration files
│   ├── model/             # Model configurations
│   ├── data/              # Data configurations
│   ├── training/          # Training configurations
│   ├── evaluation/        # Evaluation configurations
│   └── visualization/     # Visualization configurations
├── data/                  # Data directory
├── checkpoints/           # Model checkpoints
├── logs/                  # Training logs
├── assets/                # Generated visualizations
└── tests/                 # Unit tests
```

## WSOL Methods

### Grad-CAM
- **Description**: Uses gradients flowing into the final convolutional layer to produce a localization map
- **Advantages**: Simple, effective, works with any CNN
- **Use Case**: General-purpose object localization

### Grad-CAM++
- **Description**: Improved version that better captures multiple occurrences of objects
- **Advantages**: Better for multiple objects, more accurate localization
- **Use Case**: Images with multiple instances of the same object

### Score-CAM
- **Description**: Gradient-free method using forward passes with masked inputs
- **Advantages**: No gradients needed, more interpretable
- **Use Case**: When gradient information is not available or reliable

### CAM
- **Description**: Class Activation Mapping requiring global average pooling
- **Advantages**: Very fast, no backward pass needed
- **Use Case**: When architectural changes are acceptable

## Evaluation Metrics

### Classification Metrics
- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Per-class and weighted averages
- **Top-K Accuracy**: Top-1, Top-5, Top-10 accuracy

### Localization Metrics
- **Pointing Game**: Accuracy of peak attention location
- **Top-1 Localization**: IoU-based localization accuracy
- **GT-Known Localization**: Localization accuracy with ground truth class
- **Mean IoU**: Average Intersection over Union

## Configuration

The project uses Hydra for configuration management. Key configuration files:

- `configs/config.yaml`: Main configuration
- `configs/model/`: Model-specific settings
- `configs/data/`: Data loading and augmentation
- `configs/training/`: Training hyperparameters
- `configs/evaluation/`: Evaluation settings

### Example Configuration Override

```bash
python src/train.py \
    model=gradcam++ \
    data.batch_size=64 \
    training.epochs=100 \
    training.optimizer.lr=0.0001
```

## Dataset

The project includes a synthetic dataset generator for demonstration purposes. For real datasets:

1. Place images in `data/images/`
2. Create annotations in `data/annotations.csv` with columns:
   - `image_path`: Path to image file
   - `class_id`: Integer class label
   - `class_name`: String class name
   - `split`: train/val/test

## API Usage

### Basic Usage

```python
from src.models import create_model
from src.data import create_dataloaders
from src.eval import evaluate_model

# Create model
model = create_model("gradcam", num_classes=1000)

# Create data loaders
train_loader, val_loader, test_loader = create_dataloaders(config)

# Evaluate model
metrics = evaluate_model(model, test_loader, device)
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

### Custom Model

```python
from src.models import BaseWSOLModel

class CustomWSOLModel(BaseWSOLModel):
    def __init__(self, num_classes):
        super().__init__(num_classes)
        # Your implementation
    
    def forward(self, x):
        # Forward pass
        pass
    
    def get_attention_maps(self, x):
        # Attention map generation
        pass
```

## Visualization

The project provides comprehensive visualization tools:

### Attention Maps
- Overlay attention maps on original images
- Heatmap visualizations
- Side-by-side comparisons

### Training Progress
- Loss and accuracy curves
- Learning rate schedules
- Model performance comparisons

### Interactive Plots
- Plotly-based interactive visualizations
- Metrics comparison across models
- Attention map distributions

## Performance

### Model Efficiency
- **Grad-CAM**: Fastest, minimal computational overhead
- **Grad-CAM++**: Moderate overhead, better accuracy
- **Score-CAM**: Slower due to multiple forward passes
- **CAM**: Fastest but requires architectural changes

### Memory Usage
- Typical memory usage: 2-4 GB GPU memory
- Batch size: 32-64 depending on model and image size
- Mixed precision training supported

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/
ruff check src/

# Type checking
mypy src/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{wsol2024,
  title={Weakly Supervised Object Localization},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Weakly-Supervised-Object-Localization}
}
```

## Acknowledgments

- Original Grad-CAM paper: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks"
- Grad-CAM++ paper: Chattopadhyay et al., "Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks"
- Score-CAM paper: Wang et al., "Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks"
- CAM paper: Zhou et al., "Learning Deep Features for Discriminative Localization"

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use gradient accumulation
2. **Slow Training**: Enable mixed precision training or use smaller models
3. **Poor Localization**: Try different WSOL methods or adjust attention thresholds
4. **Import Errors**: Ensure all dependencies are installed and Python path is correct

### Getting Help

- Check the issues section for common problems
- Create a new issue with detailed error messages
- Include system information (OS, Python version, PyTorch version)

## Roadmap

- [ ] Support for more backbone architectures (ViT, EfficientNet)
- [ ] Additional WSOL methods (RISE, LIME, Integrated Gradients)
- [ ] Video object localization
- [ ] Multi-scale attention maps
- [ ] Attention map refinement techniques
- [ ] Integration with popular datasets (ImageNet, COCO)
- [ ] Model compression and optimization
- [ ] Web-based demo deployment
# Weakly-Supervised-Object-Localization
