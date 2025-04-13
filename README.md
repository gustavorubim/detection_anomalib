# Enhanced Anomaly Detection Pipeline

This repository contains enhanced scripts for anomaly detection using the MVTecAD dataset with optimizations for RTX 4070Ti hardware.

## Features

### 1. Performance Optimizations
- Mixed precision training (16-bit) for faster training
- Model-specific batch size optimization
- Optimized worker thread count for data loading
- Efficient backbone selection
- Parallel processing for inference
- GPU acceleration settings for both training and inference

### 2. Hyperparameter Optimization
- Comprehensive hyperparameter sweep system using Optuna
- Model-specific search spaces for each anomaly detection algorithm
- Visualization of hyperparameter importance and optimization history
- Automatic training with best parameters after optimization
- Tracking and saving of optimization results

### 3. Enhanced Evaluation Metrics
- Comprehensive classification metrics (accuracy, precision, recall, F1, ROC AUC)
- Matthews Correlation Coefficient and Cohen's Kappa
- Segmentation metrics for pixel-level evaluation (IoU, Dice coefficient)
- Image quality metrics for reconstruction-based models (SSIM, PSNR)
- Per-category performance analysis for different anomaly types

### 4. Comprehensive Reporting
- Interactive HTML report with Bootstrap styling
- Performance comparison visualizations across all models
- Timing and efficiency analysis for both training and inference
- Example visualizations for each model's detection results
- Detailed model-specific performance breakdowns

### 5. Testing and Validation
- Test pipeline to validate all enhancements
- Support for subset testing with smaller dataset samples
- Proper logging and error handling throughout the pipeline
- Configurable test options with command-line arguments

## Usage

### Training with Optimized Settings
```bash
python anomalib_training_enhanced.py
```

### Hyperparameter Optimization
```bash
python anomalib_hyperparameter_sweep.py
```

### Inference with Enhanced Evaluation
```bash
python anomalib_inference_enhanced.py
```

### Generate Enhanced Evaluation Metrics
```bash
python anomalib_enhanced_evaluation.py
```

### Create Comprehensive Report
```bash
python anomalib_comprehensive_report.py
```

### Run Full Test Pipeline
```bash
python test_pipeline.py
```

For testing with a subset of data:
```bash
python test_pipeline.py
```

For testing with the full dataset:
```bash
python test_pipeline.py --full
```

Additional options:
```bash
python test_pipeline.py --help
```

## Directory Structure

```
enhanced_scripts/
├── anomalib_training_enhanced.py      # Enhanced training script
├── anomalib_inference_enhanced.py     # Enhanced inference script
├── anomalib_hyperparameter_sweep.py   # Hyperparameter optimization
├── anomalib_enhanced_evaluation.py    # Enhanced evaluation metrics
├── anomalib_comprehensive_report.py   # Comprehensive reporting
├── test_pipeline.py                   # Testing and validation
└── README.md                          # This file
```

## Requirements

- Python 3.6+
- PyTorch
- Anomalib
- Optuna
- OpenCV
- Scikit-learn
- Matplotlib
- Seaborn
- Pandas
- Jinja2

## Notes

- All scripts are optimized for RTX 4070Ti hardware
- The pipeline supports both full dataset and subset testing
- Comprehensive logging is implemented throughout the pipeline
- Error handling is robust to ensure pipeline stability
