# Fraud Detection System for Financial & E-commerce Transactions

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
├── config/                          # YAML configs (experiment & system behavior)
│   ├── data.yaml                    # Data paths, schemas, dataset-specific options
│   ├── features.yaml                # Feature switches & definitions
│   ├── model.yaml                   # Model choices & hyperparameters
│   ├── imbalance.yaml               # SMOTE / undersampling configuration
│   ├── train.yaml                   # Training settings & evaluation metrics
│   ├── explainability.yaml          # SHAP & XAI configuration
│   └── api.yaml                     # API serving configuration
│
├── data/
│   ├── raw/                         # Original datasets (never modified)
│   ├── interim/                     # Cleaned but not fully transformed data
│   ├── processed/                   # Final model-ready datasets
│   └── external/                    # Third-party or external reference data
│
├── notebooks/
│   ├── eda_ecommerce.ipynb        # EDA for e-commerce fraud data
│   ├── eda_creditcard.ipynb       # EDA for banking fraud data
│   ├── geolocation_analysis.ipynb # IP-to-country fraud pattern analysis
│   ├── feature_engineering.ipynb  # Feature exploration & validation
│   ├── modeling.ipynb             # Model experiments & comparisons
│   └── explainability.ipynb       # SHAP analysis & interpretation
│
├── src/
│   └── fraud_detection/              # Main Python package
│       ├── __init__.py               # Marks fraud_detection as a package
│
│       ├── core/                     # Core application logic (global)
│       │   ├── __init__.py
│       │   ├── config.py             # Loads & validates YAML configuration files
│       │   └── settings.py           # Global settings, paths, env vars, seeds
│
│       ├── data/                     # Data ingestion & preprocessing
│       │   ├── __init__.py
│       │   ├── load_data.py          # Load raw CSV datasets
│       │   ├── clean.py              # Missing values, duplicates, type fixes
│       │   ├── ip_geolocation.py     # IP range → country mapping logic
│       │   ├── preprocess.py         # Scaling & categorical encoding
│       │   ├── imbalance.py          # SMOTE / undersampling utilities
│       │   └── splitter.py           # Stratified train-test splitting
│
│       ├── features/                 # Feature engineering logic
│       │   ├── __init__.py
│       │   ├── time_features.py      # hour_of_day, day_of_week, time_since_signup
│       │   ├── behavioral.py         # Velocity & frequency-based fraud features
│       │   ├── transaction.py        # Amount-based & transactional features
│       │   ├── geo_features.py       # Country risk & geo-derived features
│       │   └── feature_builder.py    # Orchestrates full feature pipeline
│
│       ├── models/                   # Modeling & evaluation
│       │   ├── __init__.py
│       │   ├── baseline.py           # Logistic Regression baseline
│       │   ├── ensemble.py           # Random Forest / XGBoost / LightGBM
│       │   ├── train.py              # End-to-end training logic
│       │   ├── tuning.py             # Hyperparameter tuning
│       │   ├── evaluate.py           # AUC-PR, F1, confusion matrix
│       │   └── predict.py            # Inference utilities
│
│       ├── explainability/            # Model explainability (XAI)
│       │   ├── __init__.py
│       │   ├── shap_analysis.py      # SHAP summary & force plots
│       │   ├── feature_importance.py # Built-in model importance
│       │   └── report.py             # Human-readable explainability reports
│
│       ├── api/                      # Model serving layer
│       │   ├── __init__.py
│       │   ├── main.py               # FastAPI application entry point
│       │   ├── schemas.py            # Pydantic request/response models
│       │   └── utils.py              # API helpers & validation
│
│       ├── pipeline/                 # Reproducible ML pipelines (DVC)
│       │   ├── __init__.py
│       │   ├── dvc_stage_data.py     # Data preparation stage
│       │   ├── dvc_stage_features.py # Feature engineering stage
│       │   ├── dvc_stage_train.py    # Model training stage
│       │   └── dvc_stage_evaluate.py # Evaluation & reporting stage
│
│       └── utils/                    # Shared utilities
│           ├── __init__.py
│           ├── project_root.py       # Reliable project root resolver
│           ├── logger.py             # Centralized logging
│           ├── metrics.py            # Custom fraud-related metrics
│           ├── constants.py          # Global constants
│           └── helpers.py            # Small reusable helpers
│
├── tests/                            # Automated tests
│   ├── test_data_cleaning.py
│   ├── test_feature_engineering.py
│   ├── test_imbalance_handling.py
│   ├── test_model_training.py
│   └── test_api.py
│
├── docker/                           # Containerization
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── start.sh
│
├── scripts/                          # CLI & automation scripts
│   ├── run_eda.py
│   ├── run_training.py
│   ├── run_explainability.py
│   └── run_api.py
│
├── mlruns/                           # MLflow experiment tracking
├── dvc.yaml                          # DVC pipeline definition
├── params.yaml                       # Central experiment parameters
├── requirements.txt                 # Python dependencies
├── pyproject.toml                   # Build & packaging config
└── README.md                        # Project overview & usage

```

## Architecture

```text
+--------------------------------------+
|  Raw Transaction Data                |
|  (E-commerce & Banking Transactions) |
|  Files: Fraud_Data.csv, creditcard.csv
|  IP Mapping: IpAddress_to_Country.csv|
|  Location: data/raw/                 |
+-------------------+------------------+
                    |
                    v
+--------------------------------------+
|  Data Loading & Validation            |
|  src/fraud_detection/data/load_data.py
|  Scripts: scripts/run_eda.py          |
|  - Load CSV datasets                 |
|  - Schema & datatype validation      |
|  - Basic sanity checks               |
+-------------------+------------------+
                    |
                    v
+--------------------------------------+
|  Data Cleaning & Preprocessing        |
|  src/fraud_detection/data/clean.py   |
|  src/fraud_detection/data/preprocess.py
|  - Handle missing values             |
|  - Remove duplicates                 |
|  - Fix datatypes & timestamps        |
|  - Encode categorical variables      |
|  - Scale numerical features          |
+-------------------+------------------+
                    |
                    v
+--------------------------------------+
|  Geolocation Enrichment               |
|  src/fraud_detection/data/ip_geolocation.py
|  Notebook: notebooks/geolocation_analysis.ipynb
|  - Convert IP to integer format      |
|  - Range-based IP → Country mapping  |
|  - Country-level fraud aggregation  |
+-------------------+------------------+
                    |
                    v
+--------------------------------------+
|  Feature Engineering                  |
|  src/fraud_detection/features/*      |
|  Notebook: notebooks/feature_engineering.ipynb
|  - Time features (hour, day, signup) |
|  - Velocity & frequency features     |
|  - Transaction amount patterns       |
|  - Geo-risk indicators               |
+-------------------+------------------+
                    |
                    v
+--------------------------------------+
|  Class Imbalance Handling             |
|  src/fraud_detection/data/imbalance.py
|  - SMOTE / undersampling (train only)|
|  - Class distribution documentation |
|  - Threshold-aware sampling          |
+-------------------+------------------+
                    |
                    v
+--------------------------------------+
|  Processed & Model-Ready Data         |
|  data/processed/*.csv                |
|  - Cleaned, feature-rich datasets    |
|  - Used consistently for training    |
|    and inference                     |
+-------------------+------------------+
                    |
                    v
+--------------------------------------+
|  Exploratory Data Analysis (EDA)      |
|  notebooks/eda_ecommerce.ipynb       |
|  notebooks/eda_creditcard.ipynb      |
|  - Fraud rate analysis               |
|  - Feature distributions             |
|  - Bivariate fraud relationships     |
|  - Outlier & pattern inspection      |
+-------------------+------------------+
                    |
                    v
+--------------------------------------+
|  Model Training & Evaluation          |
|  src/fraud_detection/models/train.py |
|  src/fraud_detection/models/evaluate.py
|  Scripts: scripts/run_training.py    |
|  - Baseline: Logistic Regression     |
|  - Ensembles: RF / XGB / LGBM        |
|  - Metrics: AUC-PR, F1, Recall       |
|  - Stratified cross-validation       |
+-------------------+------------------+
                    |
                    v
+--------------------------------------+
|  Explainability & Model Insights      |
|  src/fraud_detection/explainability/*|
|  Notebook: notebooks/explainability.ipynb
|  - SHAP global feature importance    |
|  - Local explanations (TP/FP/FN)     |
|  - Analyst-friendly explanations    |
+-------------------+------------------+
                    |
                    v
+--------------------------------------+
|  Experiment Tracking & Reproducibility|
|  MLflow (mlruns/)                    |
|  DVC (dvc.yaml, pipeline stages)     |
|  - Model versioning                  |
|  - Parameter & metric tracking       |
|  - Reproducible pipelines            |
+-------------------+------------------+
                    |
                    v
+--------------------------------------+
|  Real-Time Fraud Scoring API          |
|  src/fraud_detection/api/main.py     |
|  Scripts: scripts/run_api.py         |
|  - FastAPI prediction service        |
|  - Fraud probability & decision     |
|  - Optional explanation payload     |
+-------------------+------------------+
                    |
                    v
+--------------------------------------+
|  Business & Operational Usage         |
|  - Transaction approval/decline      |
|  - Fraud monitoring dashboards       |
|  - Analyst investigation workflows  |
|  - Continuous model improvement      |
+--------------------------------------+
```
