#!/usr/bin/env python3
"""Simple script to run the WSOL demo."""

import subprocess
import sys
from pathlib import Path


def run_demo():
    """Run the Streamlit demo."""
    print("🚀 Launching Weakly Supervised Object Localization Demo...")
    print("=" * 60)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("✅ Streamlit is installed")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
        print("✅ Streamlit installed")
    
    # Run the demo
    demo_path = Path(__file__).parent / "src" / "demo.py"
    
    if not demo_path.exists():
        print(f"❌ Demo file not found: {demo_path}")
        return False
    
    print(f"📱 Starting demo from: {demo_path}")
    print("🌐 The demo will open in your browser at http://localhost:8501")
    print("📝 Upload an image to see WSOL methods in action!")
    print("\n" + "=" * 60)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(demo_path),
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = run_demo()
    sys.exit(0 if success else 1)
