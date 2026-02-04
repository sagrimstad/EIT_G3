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
├── data/               # Raw and processed data (gitignored)
├── notebooks/          # Exploratory Data Analysis (EDA) & Prototyping
├── src/                # Production-ready Python scripts
│   ├── preprocessing.py
│   ├── train_model.py
│   └── predict.py
├── requirements.txt    # Python dependencies
└── README.md
