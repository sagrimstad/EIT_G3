# ⚡ Aneo Power Consumption Prediction
### Experts in Teams (EiT) | Village: [TDT4861 Eksperter i team - Bruk av AI og stordata for bærekraftig utvikling]

An interdisciplinary project aimed at leveraging **Aneo's weather data** to accurately forecast tomorrow's power consumption. This tool helps optimize grid management and energy distribution by anticipating demand based on environmental variables.

---

## 📖 Table of Contents
* [Overview](#-overview)
* [The Dataset](#-the-dataset)
* [Project Structure](#-project-structure)
* [Installation & Setup](#-installation--setup)
* [Workflow](#-workflow)
* [Model & Approach](#-model--approach)
* [Team (The Experts)](#-team-the-experts)

---

## 🔍 Overview
Predicting power consumption is a key challenge in the green energy transition. By analyzing historical weather patterns provided by **Aneo**—such as temperature, wind speed, and precipitation—this project builds a machine learning pipeline to predict the total load for the following day.

**Core Objectives:**
* Preprocess and align multi-source weather data.
* Engineering features (e.g., "Day of Week", "Holiday status", "Temperature lag").
* Provide a reliable prediction for tomorrow's energy demand to assist in grid balancing.

## 📊 The Dataset
The project primary utilizes:
* **Aneo Weather Data:** Hourly historical data and forecasts including temperature, wind speed, and humidity.
* **Historical Consumption Data:** Time-series power load data (measured in MWh).

> **Note:** Due to data privacy/NDA (common in EiT), raw data files are stored locally and are excluded from this repository via `.gitignore`.

## 📂 Project Structure
```text
├── data/                     # Raw and processed data (gitignored)
├── notebooks/                # EDA, prototyping, experiments
├── artifacts/                # Saved models, metadata, tuning results (gitignored)
├── outputs/                  # Predictions / evaluation outputs (gitignored)
├── scripts/                  # CLI entrypoints
│   ├── train_model.py        # Tune on validation window + train final model for one chosen test day
│   └── predict_day.py        # Load trained model and predict one chosen day
├── src/                      # Reusable production pipeline modules
│   ├── __init__.py
│   ├── utils.py              # Logging helpers
│   ├── metrics.py            # MAE / RMSE / SMAPE
│   ├── preprocessing.py      # Load, filter by location, parse timestamps
│   ├── features.py           # Feature engineering (incl. delayed consumption features)
│   ├── splits.py             # Build train/validation/test-day splits
│   ├── modeling.py           # LightGBM model creation + fit/predict helpers
│   ├── tuning.py             # Coarse + refine hyperparameter search on validation days
│   ├── train.py              # End-to-end training pipeline (tune + final retrain)
│   └── predict.py            # End-to-end prediction pipeline for one chosen day
├── requirements.txt          # Python dependencies
└── README.md
