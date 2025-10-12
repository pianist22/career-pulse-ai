#!/usr/bin/env python3
"""
NER Setup Script for Career Pulse AI
Installs required spaCy models and sets up NER environment
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False


def install_spacy_model(model_name="en_core_web_sm"):
    """Install spaCy model"""
    print(f"📦 Installing spaCy model: {model_name}")
    
    # Try to download the model
    if run_command(f"python -m spacy download {model_name}", f"Downloading {model_name}"):
        return True
    
    # If download fails, try alternative installation
    print(f"⚠️  Direct download failed. Trying alternative installation...")
    if run_command(f"pip install {model_name}", f"Installing {model_name} via pip"):
        return True
    
    print(f"❌ Could not install {model_name}")
    print("💡 Manual installation required:")
    print(f"   python -m spacy download {model_name}")
    return False


def verify_installation():
    """Verify that NER components are working"""
    print("🔍 Verifying NER installation...")
    
    try:
        import spacy
        print("✅ spaCy imported successfully")
        
        # Try to load the model
        nlp = spacy.load("en_core_web_sm")
        print("✅ spaCy model loaded successfully")
        
        # Test basic functionality
        doc = nlp("Hello world")
        print("✅ Basic NER functionality working")
        
        return True
        
    except ImportError as e:
        print(f"❌ spaCy import failed: {e}")
        return False
    except OSError as e:
        print(f"❌ spaCy model not found: {e}")
        print("💡 Run: python -m spacy download en_core_web_sm")
        return False
    except Exception as e:
        print(f"❌ NER verification failed: {e}")
        return False


def main():
    """Main setup function"""
    print("🚀 Setting up NER for Career Pulse AI")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python version: {sys.version}")
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing Python requirements"):
        print("❌ Failed to install requirements")
        sys.exit(1)
    
    # Install spaCy model
    model_name = "en_core_web_sm"
    if not install_spacy_model(model_name):
        print("⚠️  spaCy model installation failed, but continuing...")
        print("💡 You may need to install it manually later")
    
    # Verify installation
    if verify_installation():
        print("\n🎉 NER setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Run: python -m src.preprocess.normalize")
        print("2. Check the generated parquet files for entity columns")
        print("3. Run your training notebooks")
    else:
        print("\n⚠️  NER setup completed with warnings")
        print("💡 Some components may need manual installation")
    
    print("\n📚 Available spaCy models:")
    print("- en_core_web_sm (small, fast)")
    print("- en_core_web_md (medium, balanced)")
    print("- en_core_web_lg (large, most accurate)")
    print("\n💡 To install a different model:")
    print("   python -m spacy download en_core_web_md")


if __name__ == "__main__":
    main()
