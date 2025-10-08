"""
Complete Pipeline Execution Script

This script runs the entire AI Resume Shortener pipeline with NER integration:
1. Data Ingestion (ResumeAtlas + Local files)
2. NER Processing (Entity extraction)
3. Preprocessing (Text cleaning + NER features)
4. Model Training (Enhanced with NER features)
5. Evaluation and Analysis

Usage: python run_complete_pipeline.py
"""

import sys
import os
import subprocess
from pathlib import Path
import pandas as pd
import json
from datetime import datetime

# Add src to path
sys.path.append('src')

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ SUCCESS: {description}")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR: {description}")
        print(f"Error: {e.stderr}")
        return False

def check_dependencies():
    """Check if all required dependencies are installed."""
    print("🔍 CHECKING DEPENDENCIES...")
    
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'transformers', 'datasets',
        'spacy', 'nltk', 'torch', 'fitz', 'docx', 'pytesseract'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - MISSING")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {missing_packages}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed!")
    return True

def setup_environment():
    """Setup the environment and create necessary directories."""
    print("\n🔧 SETTING UP ENVIRONMENT...")
    
    # Create necessary directories
    directories = [
        "data/raw",
        "data/processed", 
        "data/interim",
        "data/processed/ner_enhanced",
        "artifacts",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Check if spaCy model is installed
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("✅ spaCy model 'en_core_web_sm' is available")
    except OSError:
        print("⚠️  spaCy model 'en_core_web_sm' not found")
        print("Installing spaCy model...")
        try:
            subprocess.run("python -m spacy download en_core_web_sm", shell=True, check=True)
            print("✅ spaCy model installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install spaCy model. Please run: python -m spacy download en_core_web_sm")
            return False
    
    return True

def run_data_ingestion():
    """Run data ingestion from ResumeAtlas and local files."""
    print("\n📥 RUNNING DATA INGESTION...")
    
    # Step 1: Load ResumeAtlas dataset
    success1 = run_command(
        "python -c \"from src.ingest.load_resumeatlas import main; main()\"",
        "Loading ResumeAtlas Dataset"
    )
    
    # Step 2: Load local files (if any exist)
    if Path("data/raw").exists() and any(Path("data/raw").iterdir()):
        success2 = run_command(
            "python -c \"from src.ingest.load_local_files import main; main()\"",
            "Loading Local Files"
        )
    else:
        print("ℹ️  No local files found in data/raw, skipping local file ingestion")
        success2 = True
    
    return success1 and success2

def run_ner_processing():
    """Run NER processing on the ingested data."""
    print("\n🧠 RUNNING NER PROCESSING...")
    
    # Check if we have data to process
    resumeatlas_file = Path("data/processed/resumeatlas_raw.parquet")
    if not resumeatlas_file.exists():
        print("❌ No ResumeAtlas data found. Please run data ingestion first.")
        return False
    
    # Run NER processing
    success = run_command(
        "python -c \"from src.preprocess.process_ner import main; main()\"",
        "NER Entity Extraction and Feature Creation"
    )
    
    return success

def run_preprocessing():
    """Run preprocessing with NER integration."""
    print("\n🔄 RUNNING PREPROCESSING WITH NER...")
    
    success = run_command(
        "python -c \"from src.preprocess.normalize import main; main()\"",
        "Text Preprocessing with NER Features"
    )
    
    return success

def run_model_training():
    """Run model training with enhanced features."""
    print("\n🤖 RUNNING MODEL TRAINING...")
    
    # Check if we have processed data
    train_file = Path("data/processed/classification_train.parquet")
    if not train_file.exists():
        print("❌ No training data found. Please run preprocessing first.")
        return False
    
    # Run training using the notebook
    success = run_command(
        "python -c \"exec(open('notebooks/01_quick_sanity_checks.ipynb').read())\"",
        "Model Training with NER Features"
    )
    
    return success

def run_evaluation():
    """Run evaluation and analysis."""
    print("\n📊 RUNNING EVALUATION AND ANALYSIS...")
    
    # Run NER integration test
    success1 = run_command(
        "python test_ner_integration.py",
        "NER Integration Testing"
    )
    
    # Run NER pipeline demonstration
    success2 = run_command(
        "python run_ner_pipeline.py",
        "NER Pipeline Demonstration"
    )
    
    return success1 and success2

def generate_summary_report():
    """Generate a summary report of the pipeline execution."""
    print("\n📋 GENERATING SUMMARY REPORT...")
    
    report = {
        "execution_time": datetime.now().isoformat(),
        "pipeline_status": "COMPLETED",
        "output_files": [],
        "ner_features": {},
        "model_performance": {}
    }
    
    # Check output files
    output_files = [
        "data/processed/resumeatlas_raw.parquet",
        "data/processed/classification_train.parquet",
        "data/processed/classification_val.parquet", 
        "data/processed/classification_test.parquet",
        "data/processed/ner_enhanced/enhanced_with_ner.parquet",
        "data/processed/ner_enhanced/extracted_entities.json",
        "data/processed/ner_enhanced/ner_features.parquet",
        "artifacts/distilbert_resume_cls_final"
    ]
    
    for file_path in output_files:
        if Path(file_path).exists():
            report["output_files"].append(file_path)
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - NOT FOUND")
    
    # Save report
    with open("logs/pipeline_execution_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Summary report saved to: logs/pipeline_execution_report.json")
    return report

def main():
    """Main function to run the complete pipeline."""
    print("🚀 STARTING AI RESUME SHORTENER PIPELINE WITH NER INTEGRATION")
    print("=" * 80)
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n❌ DEPENDENCY CHECK FAILED")
        print("Please install missing dependencies and try again.")
        return False
    
    # Step 2: Setup environment
    if not setup_environment():
        print("\n❌ ENVIRONMENT SETUP FAILED")
        return False
    
    # Step 3: Run data ingestion
    if not run_data_ingestion():
        print("\n❌ DATA INGESTION FAILED")
        return False
    
    # Step 4: Run NER processing
    if not run_ner_processing():
        print("\n❌ NER PROCESSING FAILED")
        return False
    
    # Step 5: Run preprocessing
    if not run_preprocessing():
        print("\n❌ PREPROCESSING FAILED")
        return False
    
    # Step 6: Run model training
    if not run_model_training():
        print("\n❌ MODEL TRAINING FAILED")
        return False
    
    # Step 7: Run evaluation
    if not run_evaluation():
        print("\n❌ EVALUATION FAILED")
        return False
    
    # Step 8: Generate summary report
    report = generate_summary_report()
    
    print("\n" + "=" * 80)
    print("🎉 PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    print("\n📁 OUTPUT FILES CREATED:")
    for file_path in report["output_files"]:
        print(f"  ✅ {file_path}")
    
    print("\n🔍 NEXT STEPS:")
    print("1. Check the generated files in data/processed/")
    print("2. Review NER analysis in data/processed/ner_enhanced/")
    print("3. Examine model artifacts in artifacts/")
    print("4. Check logs for detailed execution information")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ COMPLETE PIPELINE EXECUTION SUCCESSFUL!")
            sys.exit(0)
        else:
            print("\n❌ PIPELINE EXECUTION FAILED!")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        sys.exit(1)
