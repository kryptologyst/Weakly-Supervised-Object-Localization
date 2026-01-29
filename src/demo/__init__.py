"""Demo interface for weakly supervised object localization."""

import os
from typing import Dict, List, Optional, Tuple

import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

from .models import create_model
from .utils import get_device, denormalize_image


class WSOLDemo:
    """Demo interface for WSOL models."""
    
    def __init__(self) -> None:
        """Initialize demo."""
        self.device = get_device("auto")
        self.model = None
        self.class_names = [
            "tench", "goldfish", "great white shark", "tiger shark", "hammerhead",
            "electric ray", "stingray", "cock", "hen", "ostrich"
        ]
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    def load_model(self, model_name: str) -> None:
        """Load a WSOL model.
        
        Args:
            model_name: Name of the model to load.
        """
        self.model = create_model(model_name, num_classes=len(self.class_names))
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, image: Image.Image) -> Tuple[str, float, np.ndarray]:
        """Make prediction on an image.
        
        Args:
            image: Input image.
            
        Returns:
            Tuple of (predicted_class, confidence, attention_map).
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load a model first.")
        
        # Preprocess image
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get prediction
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
            # Get attention map
            attention_map = self.model.get_attention_maps(input_tensor)
            attention_map_np = attention_map.cpu().numpy()[0, 0]
        
        return (
            self.class_names[predicted_class.item()],
            confidence.item(),
            attention_map_np
        )
    
    def create_visualization(
        self,
        image: Image.Image,
        predicted_class: str,
        confidence: float,
        attention_map: np.ndarray,
    ) -> Image.Image:
        """Create visualization of prediction and attention map.
        
        Args:
            image: Original image.
            predicted_class: Predicted class name.
            confidence: Prediction confidence.
            attention_map: Attention map.
            
        Returns:
            Visualization image.
        """
        # Resize image to match attention map
        image_resized = image.resize((224, 224))
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image_resized)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        
        # Attention map
        im = axes[1].imshow(attention_map, cmap="jet")
        axes[1].set_title("Attention Map")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1])
        
        # Overlay
        axes[2].imshow(image_resized)
        axes[2].imshow(attention_map, alpha=0.6, cmap="jet")
        axes[2].set_title(f"Prediction: {predicted_class}\nConfidence: {confidence:.3f}")
        axes[2].axis("off")
        
        plt.tight_layout()
        
        # Convert to PIL Image
        fig.canvas.draw()
        vis_image = Image.frombytes(
            "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
        )
        
        plt.close(fig)
        return vis_image


def run_demo() -> None:
    """Run the Streamlit demo."""
    st.set_page_config(
        page_title="Weakly Supervised Object Localization Demo",
        page_icon="🔍",
        layout="wide",
    )
    
    st.title("🔍 Weakly Supervised Object Localization Demo")
    st.markdown(
        "Upload an image to see how different WSOL methods localize objects using only class labels."
    )
    
    # Initialize demo
    if "demo" not in st.session_state:
        st.session_state.demo = WSOLDemo()
    
    demo = st.session_state.demo
    
    # Sidebar for model selection
    st.sidebar.header("Model Selection")
    model_name = st.sidebar.selectbox(
        "Choose WSOL Method:",
        ["gradcam", "gradcam++", "scorecam", "cam"],
        help="Select the weakly supervised object localization method to use."
    )
    
    # Load model
    if st.sidebar.button("Load Model"):
        with st.spinner("Loading model..."):
            demo.load_model(model_name)
        st.sidebar.success(f"Loaded {model_name.upper()} model!")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Input Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png"],
            help="Upload an image to analyze with WSOL methods."
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Process image
            if demo.model is not None:
                with st.spinner("Processing image..."):
                    predicted_class, confidence, attention_map = demo.predict(image)
                
                # Create visualization
                vis_image = demo.create_visualization(
                    image, predicted_class, confidence, attention_map
                )
                
                with col2:
                    st.header("Results")
                    st.image(vis_image, caption="WSOL Analysis", use_column_width=True)
                    
                    # Display metrics
                    st.subheader("Prediction Details")
                    st.write(f"**Predicted Class:** {predicted_class}")
                    st.write(f"**Confidence:** {confidence:.3f}")
                    st.write(f"**Method:** {model_name.upper()}")
                    
                    # Attention map statistics
                    st.subheader("Attention Map Statistics")
                    st.write(f"**Max Value:** {attention_map.max():.3f}")
                    st.write(f"**Min Value:** {attention_map.min():.3f}")
                    st.write(f"**Mean Value:** {attention_map.mean():.3f}")
                    st.write(f"**Std Value:** {attention_map.std():.3f}")
                    
                    # Attention map histogram
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.hist(attention_map.flatten(), bins=50, alpha=0.7)
                    ax.set_xlabel("Attention Value")
                    ax.set_ylabel("Frequency")
                    ax.set_title("Attention Map Distribution")
                    st.pyplot(fig)
            else:
                st.warning("Please load a model first using the sidebar.")
        else:
            st.info("Please upload an image to get started.")
    
    # Information section
    st.header("About WSOL Methods")
    
    method_info = {
        "gradcam": {
            "name": "Grad-CAM",
            "description": "Uses gradients flowing into the final convolutional layer to produce a localization map.",
            "pros": ["Simple and effective", "Works with any CNN", "No architectural changes needed"],
            "cons": ["Requires gradients", "May miss fine details", "Limited to final conv layer"]
        },
        "gradcam++": {
            "name": "Grad-CAM++",
            "description": "Improved version of Grad-CAM that better captures multiple occurrences of objects.",
            "pros": ["Better for multiple objects", "More accurate localization", "Improved weights calculation"],
            "cons": ["More complex", "Still gradient-dependent", "Computationally heavier"]
        },
        "scorecam": {
            "name": "Score-CAM",
            "description": "Gradient-free method that uses forward passes with masked inputs to generate explanations.",
            "pros": ["No gradients needed", "More interpretable", "Works with any model"],
            "cons": ["Computationally expensive", "Many forward passes", "May be slow"]
        },
        "cam": {
            "name": "CAM",
            "description": "Class Activation Mapping that requires global average pooling and linear layers.",
            "pros": ["Very fast", "No backward pass", "Clear interpretation"],
            "cons": ["Requires specific architecture", "Limited flexibility", "Needs architectural changes"]
        }
    }
    
    selected_info = method_info.get(model_name, method_info["gradcam"])
    
    st.subheader(f"{selected_info['name']} Details")
    st.write(selected_info["description"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Advantages:**")
        for pro in selected_info["pros"]:
            st.write(f"• {pro}")
    
    with col2:
        st.write("**Limitations:**")
        for con in selected_info["cons"]:
            st.write(f"• {con}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "This demo showcases different weakly supervised object localization methods. "
        "Upload an image and try different methods to see how they localize objects!"
    )


if __name__ == "__main__":
    run_demo()
