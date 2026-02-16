# Fraud Detection System for Financial & E-commerce Transactions
![CI](https://github.com/tib-dev/fraud-detection/actions/workflows/ci.yml/badge.svg)

An end-to-end, production-ready **fraud detection platform** designed to identify suspicious transactions in real time.  
The system transforms raw transactional and behavioral data into fraud signals, trains and tracks machine learning models, and exposes predictions through a scalable API.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Business Context](#business-context)
- [Fraud Detection Business Understanding](#fraud-detection-business-understanding)
- [Objectives](#objectives)
- [Dataset Overview](#dataset-overview)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Modeling Approach](#modeling-approach)
- [MLOps & Engineering Practices](#mlops--engineering-practices)
- [Setup & Installation](#setup--installation)
- [Running the Project](#running-the-project)
- [API Usage](#api-usage)
- [Technologies Used](#technologies-used)
- [Author](#author)

---

## Project Overview

This project implements a **Fraud Detection System** for financial and e-commerce transactions.  
It is designed to detect fraudulent behavior in environments where transaction volume is high, patterns evolve quickly, and false positives are costly.

The system covers the full lifecycle:

- Data ingestion and cleaning
- Feature engineering (temporal, behavioral, transactional, geolocation)
- Class imbalance handling (SMOTE / undersampling)
- Model training, tuning, and evaluation
- Explainability with SHAP and feature importance
- Experiment tracking and reproducibility
- Real-time fraud prediction via FastAPI
- Containerized deployment using Docker

---

## Business Context

Financial institutions and e-commerce platforms face continuous losses due to fraudulent transactions.  
Fraud patterns change rapidly, making rule-based systems brittle and hard to maintain.

This project supports:

- **Real-time fraud screening**
- **Risk-based transaction approval**
- **Fraud monitoring and investigation**
- **Reduction of false positives**

The output is a **fraud probability score** that can be integrated into transaction decision pipelines.

---

## Fraud Detection Business Understanding

### Nature of Fraud Problems

Fraud detection differs from traditional classification tasks:

- Fraud cases are **rare** (high class imbalance)
- Fraud patterns **evolve over time**
- False positives directly impact customer experience
- Explainability is required for investigations and audits

This system is designed with these constraints in mind.

### Class Imbalance Strategy

Fraud labels are highly imbalanced.  
To address this, the project supports:

- SMOTE and controlled oversampling
- Undersampling of majority class
- Threshold tuning based on business cost
- Precision-Recall focused evaluation (AUC-PR, F1)

### Model Interpretability

Fraud predictions must be explainable for:

- Internal fraud analysts
- Customer dispute resolution
- Regulatory and audit requirements

This project integrates:

- SHAP value analysis
- Global and local feature importance
- Human-readable explainability reports

---

## Objectives

- Detect fraudulent transactions accurately
- Engineer robust fraud-specific features
- Handle extreme class imbalance safely
- Compare interpretable and complex models
- Explain model predictions
- Serve predictions through a scalable API
- Ensure reproducibility and auditability

---

## Dataset Overview

**Sources:**

- E-commerce transaction logs
- Banking / credit card transaction datasets
- IP geolocation reference data

**Key fields include:**

| Column               | Description                        |
| -------------------- | ---------------------------------- |
| TransactionId        | Unique transaction identifier      |
| CustomerId           | Unique customer identifier         |
| Amount               | Transaction amount                 |
| ChannelId            | Platform channel                   |
| TransactionStartTime | Timestamp                          |
| IPAddress            | Client IP address                  |
| FraudResult          | Fraud label (0 = legit, 1 = fraud) |

Derived features include:

- Time-based signals
- Velocity and frequency features
- Amount-based aggregations
- Country and geolocation risk indicators

---

## Project Structure

```text
fraud-detection/
│
├── config/                          # YAML configuration files for system behavior and experiments
│   ├── paths.yaml                    # Data paths, raw/interim/processed directories
│   ├── features.yaml                 # Feature switches, definitions, and pipelines
│   ├── profile.yaml                  # Training settings, evaluation metrics, hyperparameters
│   └── api.yaml                      # API serving configuration
│
├── data/                             # Storage for all dataset versions
│   ├── raw/                          # Original datasets (never modified)
│   ├── interim/                      # Cleaned but not fully transformed data
│   ├── processed/                    # Final model-ready datasets for training and inference
│   └── external/                     # Third-party or reference data (e.g., IP mapping)
│
├── notebooks/                        # Jupyter notebooks for EDA, feature engineering, modeling
│   ├── eda_ecommerce.ipynb                   # E-commerce fraud exploratory analysis
│   ├── credit_card_eda_feature_engineering.ipynb  # Credit card fraud EDA and feature engineering
│   ├── credit_card_fraud_modeling.ipynb     # Modeling experiments for credit card fraud
│   ├── ecommerce_fraud_modeling.ipynb       # Modeling experiments for e-commerce fraud
│   ├── ecommerce_eda_feature_engineering.ipynb   # Feature exploration and engineering for e-commerce
│   └── explainability.ipynb                 # SHAP & local/global explanations of trained models
│
├── src/
│   └── fraud_detection/               # Main Python package
│       ├── __init__.py                # Marks fraud_detection as a package
│
│       ├── analysis/                  # Exploratory data analysis tools
│       │   ├── __init__.py
│       │   └── eda.py                 # Functions for EDA and visualization of fraud patterns
│
│       ├── api/                       # FastAPI serving layer
│       │   ├── __init__.py
│       │   ├── main.py                # API entry point (prediction service)
│       │   ├── schemas.py             # Request/response Pydantic models
│       │   └── utils.py               # Helper functions for API operations
│
│       ├── core/                      # Global core functionality and configuration
│       │   ├── __init__.py
│       │   └── settings.py            # Loads settings and global constants
│
│       ├── data/                      # Data ingestion and preprocessing
│       │   ├── __init__.py
│       │   ├── cleaning.py            # Missing values, duplicates, type fixes
│       │   ├── ip_geolocation.py      # IP → Country mapping logic
│       │   └── loader.py              # CSV/parquet dataset loading utilities
│
│       ├── explainability/            # Model explainability and interpretation
│       │   ├── __init__.py
│       │   ├── feature_importance.py  # Built-in and SHAP feature importance extraction
│       │   ├── predictions.py         # Prediction-level explanations (TP, FP, FN)
│       │   └── shap_explainer.py      # SHAP TreeExplainer utilities and single prediction analysis
│
│       ├── features/                  # Feature engineering modules
│       │   ├── __init__.py
│       │   ├── custom_features.py     # Domain-specific features
│       │   ├── pipeline.py            # Orchestrates full feature pipeline
│       │   └── preprocessing.py       # Scaling, encoding, and transformations
│
│       ├── models/                    # Model training, evaluation, and persistence
│       │   ├── __init__.py
│       │   ├── compare.py             # Model comparison utilities
│       │   ├── metrics.py             # Evaluation metrics for fraud detection
│       │   ├── persistence.py         # Model saving/loading
│       │   ├── pipeline.py            # ML pipeline utilities
│       │   ├── tracker.py             # Experiment tracking & logging
│       │   ├── train.py               # End-to-end training scripts
│       │   └── tuning.py              # Hyperparameter tuning routines
│
│       ├── utils/                      # Shared helper functions
│       │   ├── __init__.py
│       │   ├── helpers.py             # Reusable utility functions
│       │   ├── mlflow_tracking.py     # MLflow experiment tracking helpers
│       │   └── project_root.py        # Resolving project root reliably
│
│       └── viz/                        # Visualization utilities
│           ├── __init__.py
│           ├── importance_plots.py    # Feature importance plots
│           ├── model_plots.py         # Model-specific plots (ROC, confusion matrix)
│           └── plots.py               # Generic visualization utilities
│
├── tests/                                # Automated testing
│   ├── test_data_cleaning.py
│   ├── test_feature_engineering.py
│   ├── test_imbalance_handling.py
│   ├── test_model_training.py
│   └── test_api.py
│
├── docker/                               # Containerization & deployment
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── start.sh
│
├── scripts/                              # CLI & automation entry points
│   ├── run_pipeline.py
│   ├── run_ecommerce_training.py
│   ├── run_explainability.py
│   └── run_api.py
│
├── mlruns/                               # MLflow experiment tracking
├── dvc.yaml                              # DVC pipeline definition
├── params.yaml                           # Global experiment parameters
├── requirements.txt                     # Runtime dependencies
├── pyproject.toml                       # Build & packaging configuration
└── README.md                            # Project overview & usage



## Architecture
```text
+--------------------------------------------------+
|                Fraud Detection System            |
|        End-to-End ML + Explainability Platform   |
+--------------------------------------------------+

+--------------------------------------------------+
|  Raw Transaction & Reference Data                |
|--------------------------------------------------|
|  - E-commerce / Banking transactions             |
|  - IP → Country mapping tables                   |
|  Location: data/raw/                             |
+------------------------+-------------------------+
                         |
                         v
+--------------------------------------------------+
|  Data Access & Loading                            |
|--------------------------------------------------|
|  src/fraud_detection/data/loader.py               |
|  - Centralized dataset loading                   |
|  - Parquet / CSV abstraction                     |
|  - Schema-aware reads                             |
+------------------------+-------------------------+
                         |
                         v
+--------------------------------------------------+
|  Data Cleaning & Validation                       |
|--------------------------------------------------|
|  src/fraud_detection/data/cleaning.py             |
|  - Missing value handling                         |
|  - Duplicate removal                              |
|  - Type normalization                             |
|  - Timestamp coercion                             |
+------------------------+-------------------------+
                         |
                         v
+--------------------------------------------------+
|  IP Geolocation Enrichment                        |
|--------------------------------------------------|
|  src/fraud_detection/data/ip_geolocation.py       |
|  - IP → integer conversion                        |
|  - Range-based country mapping                    |
|  - Safe merge (merge_asof)                        |
|  - Unknown country handling                       |
+------------------------+-------------------------+
                         |
                         v
+--------------------------------------------------+
|  Feature Engineering Pipeline                     |
|--------------------------------------------------|
|  src/fraud_detection/features/                    |
|    ├─ custom_features.py                          |
|    │   - Time features (hour, weekday)            |
|    │   - Velocity features (1h / 24h counts)      |
|    │   - Device & IP frequency                    |
|    ├─ preprocessing.py                            |
|    │   - Encoding                                 |
|    │   - Scaling                                  |
|    └─ pipeline.py                                 |
|        - sklearn-compatible feature pipeline      |
+------------------------+-------------------------+
                         |
                         v
+--------------------------------------------------+
|  Exploratory Data Analysis (EDA)                  |
|--------------------------------------------------|
|  src/fraud_detection/analysis/eda.py              |
|  - Class imbalance analysis                       |
|  - Feature distribution inspection                |
|  - Fraud pattern exploration                      |
|  - Business-oriented insights                     |
+------------------------+-------------------------+
                         |
                         v
+--------------------------------------------------+
|  Model Training & Tuning                          |
|--------------------------------------------------|
|  src/fraud_detection/models/                      |
|    ├─ train.py                                    |
|    │   - End-to-end training pipeline              |
|    ├─ tuning.py                                   |
|    │   - Hyperparameter optimization               |
|    ├─ compare.py                                  |
|    │   - Model benchmarking                        |
|    ├─ metrics.py                                  |
|    │   - Recall-focused metrics (AUC-PR, Recall)  |
|    ├─ pipeline.py                                 |
|    │   - Full sklearn pipeline                     |
|    └─ persistence.py                              |
|        - Model save/load                           |
+------------------------+-------------------------+
                         |
                         v
+--------------------------------------------------+
|  Experiment Tracking & Reproducibility            |
|--------------------------------------------------|
|  src/fraud_detection/models/tracker.py            |
|  src/fraud_detection/utils/mlflow_tracking.py     |
|  - MLflow experiment tracking                     |
|  - Parameter & metric logging                     |
|  - Model registry integration                     |
+------------------------+-------------------------+
                         |
                         v
+--------------------------------------------------+
|  Explainability & Model Insights                  |
|--------------------------------------------------|
|  src/fraud_detection/explainability/              |
|    ├─ feature_importance.py                       |
|    │   - Built-in feature importance               |
|    ├─ shap_explainer.py                           |
|    │   - SHAP value computation                    |
|    │   - Global & local explanations               |
|    └─ predictions.py                              |
|        - TP / FP / FN case extraction              |
+------------------------+-------------------------+
                         |
                         v
+--------------------------------------------------+
|  Visualization Layer                              |
|--------------------------------------------------|
|  src/fraud_detection/viz/                         |
|    ├─ plots.py                                    |
|    │   - EDA & distribution plots                 |
|    ├─ model_plots.py                              |
|    │   - ROC, PR, confusion matrix                |
|    └─ importance_plots.py                         |
|        - Feature & SHAP bar plots                 |
+------------------------+-------------------------+
                         |
                         v
+--------------------------------------------------+
|  Inference & API Layer                            |
|--------------------------------------------------|
|  src/fraud_detection/api/                         |
|    ├─ main.py                                     |
|    │   - FastAPI entry point                      |
|    ├─ schemas.py                                  |
|    │   - Request / response models                |
|    └─ utils.py                                    |
|        - Input validation & helpers               |
+------------------------+-------------------------+
                         |
                         v
+--------------------------------------------------+
|  Configuration & Core Utilities                   |
|--------------------------------------------------|
|  src/fraud_detection/core/settings.py             |
|  src/fraud_detection/utils/helpers.py             |
|  src/fraud_detection/utils/project_root.py        |
|  - Centralized configuration                      |
|  - Path resolution                                |
|  - Shared helpers                                 |
+------------------------+-------------------------+
                         |
                         v
+--------------------------------------------------+
|  Testing & Quality Assurance                      |
|--------------------------------------------------|
|  tests/                                           |
|    ├─ test_api.py                                 |
|    ├─ test_data_handler.py                        |
|    ├─ test_feature.py                             |
|    ├─ test_feature_importance.py                  |
|    └─ test_model_training.py                      |
|  - Unit tests                                     |
|  - Pipeline regression tests                      |
|  - Explainability correctness                     |
+--------------------------------------------------+
```

## Setup & Installation

Clone the repository:

```bash
git clone https://github.com/<tib-dev>/fraud_detection.git
cd fraud_detection
```
### Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```
Install dependencies:
 - pyproject run

```bash

pip install -r requirements.txt
```
 - pyproject run
 
```bash

pip install -e .
```
## Running the Project

### Run the full pipeline (DVC)
```bash
dvc repro
```

