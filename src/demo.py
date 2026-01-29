"""Main demo script for weakly supervised object localization."""

import hydra
from omegaconf import DictConfig
import streamlit as st
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.demo import run_demo


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    """Main demo function."""
    # Set Streamlit config
    st.set_page_config(
        page_title="Weakly Supervised Object Localization Demo",
        page_icon="🔍",
        layout="wide",
    )
    
    # Run the demo
    run_demo()


if __name__ == "__main__":
    main()
