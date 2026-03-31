# Credit Risk Prediction

A machine learning pipeline for predicting credit default risk. Covers the full ML workflow — data ingestion, feature engineering, model training, and evaluation — with a clean, modular architecture.

## Features

- **Data loading** — standardised ingestion and preprocessing pipeline
- **Feature engineering** — encoding, scaling, interaction features, and missing value handling
- **Model training** — multiple classifiers with cross-validation and hyperparameter tuning
- **Evaluation** — AUC-ROC, precision-recall, confusion matrix, and feature importance

## Project Structure

```
├── main.py                # Entry point — orchestrates full pipeline
├── data_loader.py         # Load and validate raw credit data
├── feature_engineering.py # Feature transformations and selection
├── model_trainer.py       # Train and tune classifiers
├── model_evaluator.py     # Metrics, plots, and model comparison
└── __init__.py
```

## Getting Started

```bash
# Install dependencies
pip install scikit-learn pandas numpy matplotlib seaborn

# Run the pipeline
python main.py
```

## Tech Stack

- **Language**: Python 3
- **Libraries**: scikit-learn, pandas, numpy, matplotlib, seaborn
