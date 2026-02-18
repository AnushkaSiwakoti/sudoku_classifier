#!/usr/bin/env python3
"""
Run complete Sudoku Classifier ML Pipeline
==========================================

This script executes the full pipeline:
1. Generate synthetic Sudoku data
2. Extract features from raw puzzles
3. Train logistic regression classifier
4. Evaluate model and generate reports

Usage:
    python run_pipeline.py
"""
import os
import sys
import subprocess
from pathlib import Path


def print_header(title):
    """Print formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def run_step(script_path, description):
    """Run a pipeline step and handle errors."""
    print(f"🚀 {description}...")
    print(f"   Running: {script_path}")
    print("-" * 70)
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"✅ {description} completed successfully!\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {description}")
        print(f"   Exit code: {e.returncode}")
        return False


def check_dependencies():
    """Check if required packages are installed."""
    print_header("Checking Dependencies")
    
    required_packages = [
        'numpy',
        'pandas', 
        'sklearn',
        'matplotlib',
        'seaborn'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (MISSING)")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt --break-system-packages")
        return False
    
    print("\n✅ All dependencies installed!")
    return True


def create_directories():
    """Create necessary directories."""
    print_header("Setting Up Project Structure")
    
    directories = ['data', 'models', 'results', 'notebooks']
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✓ Created/verified: {dir_name}/")
    
    print("\n✅ Directory structure ready!")


def main():
    """Execute complete ML pipeline."""
    print("\n" + "🎯" * 35)
    print("     SUDOKU DIFFICULTY CLASSIFIER - ML PIPELINE")
    print("🎯" * 35)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Define pipeline steps
    steps = [
        ('src/generate_data.py', 'Step 1: Generate Sudoku Dataset'),
        ('src/feature_engineering.py', 'Step 2: Feature Engineering'),
        ('src/train_model.py', 'Step 3: Train Model & Evaluate'),
    ]
    
    # Execute pipeline
    print_header("Executing ML Pipeline")
    
    for step_num, (script, description) in enumerate(steps, 1):
        if not run_step(script, description):
            print(f"\n❌ Pipeline failed at step {step_num}")
            sys.exit(1)
    
    # Pipeline complete
    print_header("Pipeline Completed Successfully! 🎉")
    
    print("📊 Results Summary:")
    print("   • Training data: data/train_features.csv")
    print("   • Test data: data/test_features.csv")
    print("   • Trained model: models/sudoku_classifier.pkl")
    print("   • Confusion matrix: results/confusion_matrix.png")
    print("   • Feature importance: results/feature_importance.png")
    print("   • Metrics: results/metrics.csv")
    
    print("\n📓 Next Steps:")
    print("   1. Review results in results/ folder")
    print("   2. Explore data with: jupyter notebook notebooks/01_exploratory_data_analysis.ipynb")
    print("   3. Load model and make predictions (see README.md)")
    
    print("\n" + "="*70)
    print("  Thank you for running the Sudoku Classifier pipeline!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
