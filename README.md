# eXplainability for Signature Detection

A comprehensive computer vision project that combines signature classification with explainable AI (XAI) techniques for enhanced model interpretability. This project implements a deep learning pipeline for signature authentication using the CEDAR dataset, featuring modular architecture, comprehensive data analysis, and explainability tools.

## 🎯 Project Overview

This project serves as the capstone for the Professional AI Master's program in AI Engineering. It demonstrates the integration of modern deep learning techniques with explainable AI methods to create an interpretable signature classification system.

### Key Features

- **Signature Classification**: Binary classification for signature authentication using the CEDAR dataset
- **Explainable AI**: Integration of XAI techniques (Grad-CAM, LIME, etc.) for model interpretability
- **Modular Architecture**: Clean, extensible codebase with separation of concerns
- **Comprehensive EDA**: Detailed exploratory data analysis with automated reporting
- **Model Factory**: Support for multiple architectures (CNN, ResNet50, etc.)
- **Configuration-Driven**: YAML-based configuration management
- **Robust Data Pipeline**: Automated data loading, preprocessing, and augmentation

## 🏗️ Project Structure

```
eXplainability-for-signature-detection/
├── config/
│   └── config.yaml              # Project configuration
├── metadata/
│   └── metadata.json           # Dataset metadata
├── src/
│   ├── data/
│   │   └── cedar_dataset.py    # CEDAR dataset loader
│   ├── model/
│   │   ├── base.py            # Base model class
│   │   ├── cnn.py             # CNN implementations
│   │   └── model_factory.py   # Model factory
│   ├── training/
│   │   ├── callbacks.py       # Training callbacks
│   │   ├── experiment.py      # Experiment management
│   │   └── trainer.py         # Training logic
│   ├── utils/
│   │   ├── dataset_analyzer.py # Dataset analysis tools
│   │   ├── eda.py             # Exploratory data analysis
│   │   ├── kaggle_downloader.py # Data downloading
│   │   └── logger_setup.py    # Logging configuration
│   ├── visualization/         # Visualization utilities
│   └── xai/                  # Explainable AI implementations
├── XAI_project_notebook.ipynb  # Main project notebook
├── LICENSE                     # Apache 2.0 License
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- torchvision
- PIL (Pillow)
- NumPy
- Pandas
- Matplotlib
- PyYAML
- Kaggle API

### Installation

1. Clone the repository:
```bash
git clone https://github.com/silvano315/eXplainability-for-signature-detection.git
cd eXplainability-for-signature-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt  # Create this file based on your dependencies
```

3. Set up Kaggle API credentials:
   - Place your `kaggle.json` file in `~/.kaggle/`
   - Ensure proper permissions: `chmod 600 ~/.kaggle/kaggle.json`

### Usage

1. **Configuration**: Modify `config/config.yaml` to adjust training parameters, model settings, and data paths.

2. **Data Preparation**: The dataset will be automatically downloaded from Kaggle:
```python
from src.utils.kaggle_downloader import setup_dataset
dataset_path = setup_dataset()
```

3. **Exploratory Data Analysis**: Generate comprehensive EDA reports:
```python
from src.utils.eda import generate_eda_report
generate_eda_report(signatures_path, metadata, output_dir)
```

4. **Model Training**: Use the modular training pipeline:
```python
from src.training.trainer import Trainer
from src.model.model_factory import create_model

model = create_model(config['model'])
trainer = Trainer(model, config)
trainer.train(dataloaders)
```

5. **Explainability Analysis**: Apply XAI techniques to understand model decisions.

## 📊 Dataset

The project uses the **CEDAR signature dataset** from Kaggle:
- **Source**: `shreelakshmigp/cedardataset`
- **Classes**: 55 different signers
- **Task**: Binary classification (genuine vs. forged signatures)
- **Image Format**: Grayscale, normalized to 224x224 pixels

### Data Processing Features

- Automated metadata generation and validation
- Balanced train/validation/test splits
- Comprehensive data augmentation
- Class distribution analysis
- Image property analysis

## 🤖 Models

The project supports multiple model architectures through a factory pattern:

- **BaselineCNN**: Custom CNN architecture with batch normalization
- **ResNet50**: Pre-trained ResNet50 with transfer learning
- **Configurable**: Easy addition of new architectures

### Model Configuration

```yaml
model:
  name: "resnet50"
  pretrained: true
  num_classes: 2
  dropout_rate: 0.5
```

## 🔍 Explainable AI

Integration of various XAI techniques for model interpretability:

- **Grad-CAM**: Gradient-based class activation mapping
- **LIME**: Local interpretable model-agnostic explanations
- **Integrated Gradients**: Attribution-based explanations
- **Saliency Maps**: Input gradient visualizations

## 📈 Training Features

- **Configurable Training**: YAML-based parameter management
- **Advanced Optimization**: Support for various optimizers and schedulers
- **Early Stopping**: Automatic training termination based on validation metrics
- **Comprehensive Logging**: Detailed logging with configurable levels
- **Experiment Tracking**: Systematic experiment management

## 🧪 Evaluation & Analysis

- Automated model evaluation metrics
- Confusion matrix generation
- ROC curve analysis
- Comprehensive performance reporting
- Visual analysis of predictions

## 📋 Configuration

The project uses a comprehensive YAML configuration system:

```yaml
dataset:
  num_classes: 55
  train_test_split: 0.2
  validation_split: 0.1

training:
  batch_size: 32
  num_epochs: 50
  learning_rate: 0.001
  optimizer: "adam"

preprocessing:
  image:
    size: [224, 224]
    normalize: true
    channels: 1
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **CEDAR Dataset**: Thanks to the creators of the CEDAR signature dataset
- **Profession AI Master's Program**: AI Engineering track
- **PyTorch Community**: For the excellent deep learning framework
- **Kaggle**: For hosting the dataset and providing the API

---
