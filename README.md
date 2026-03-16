## Big Data — T

| Name | NRP |
|---|---|
| Dustin Felix | 5025231046 |
| Darryl Matthew Wibawa | 5025231047 |
| Nicholas | 5025231031 |
| Athalla Abhinaya | 5025231107 |
---

# ⚽ Soccer Player Performance Prediction

A machine learning project that predicts soccer player performance ratings (**Bad**, **Normal**, **Good**) based on in-game statistics. Built for Big Data coursework using Gradient Boosting with an interactive Gradio web interface.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Feature Engineering](#feature-engineering)
- [Model Selection](#model-selection)
- [Results](#results)
- [How to Run](#how-to-run)
- [Web Application](#web-application)
- [Technologies Used](#technologies-used)

---

## Overview

This project builds a classification model to predict a soccer player's performance category based on their match statistics such as goals, assists, tackles, passes, and more. The model supports four player positions: **Striker**, **Midfielder**, **Defender**, and **Goalkeeper**.

**Performance Classes:**
| Label | Meaning |
|---|---|
| `Good` | Above-average performance |
| `Normal` | Average performance |
| `Bad` | Below-average performance |
| `-` | No appearances (not rated) |

---

## Dataset

| File | Description |
|---|---|
| `soccer_player_performances.csv` | Training dataset with labeled performance |
| `soccer_player_test.csv` | Test dataset for generating predictions |
| `soccer_player_test_with_predictions.csv` | Output file with model predictions (auto-generated) |

**Key Features in the Dataset:**
- Player info: `Name`, `Age`, `Club`, `Nationality`, `Position`
- Attacking: `Goals`, `Shots`, `Shots on Target`, `Assist`, `Key Passes`
- Defensive: `Tackle Attempt`, `Tackle Won`, `Interception`, `Conceded`, `Shutouts`
- Passing: `Pass Attempt / 90 minutes`, `Pass Completed / 90 minutes`
- Physical: `Distance / 90 minutes`, `Appearances`

---

## Project Structure

```
week-3-playersoccer/
│
├── soccer_analysis.ipynb              # Main notebook (EDA + Training + Analysis)
├── app.py                             # Gradio web application for live prediction
│
├── soccer_player_performances.csv     # Training dataset
├── soccer_player_test.csv             # Test dataset
│
├── .gitignore                         # Git ignore rules
└── README.md                          # This file

# Generated locally (not pushed to GitHub):
# ├── soccer_pipeline.pkl              # Trained Gradient Boosting model
# ├── position_encoder.pkl             # Label encoder for Position column
# └── soccer_player_test_with_predictions.csv  # Prediction output
```

---

## Machine Learning Pipeline

The full pipeline inside `soccer_analysis.ipynb`:

```
Raw CSV Data
    │
    ▼
Data Cleaning  (remove rows where Performance == '-')
    │
    ▼
Feature Engineering  (add ratio-based features)
    │
    ▼
Label Encoding  (Position → numeric)
    │
    ▼
Train/Validation Split  (80% train, 20% val, stratified)
    │
    ▼
Baseline Model: Random Forest  ──────┐
    │                                │ Compare accuracy
Final Model: Gradient Boosting ──────┘
    │
    ▼
Hyperparameter Tuning (GridSearchCV + StratifiedKFold)
    │
    ▼
Evaluation: Accuracy, Classification Report, Confusion Matrix
    │
    ▼
Retrain on Full Dataset → Save Model (.pkl)
    │
    ▼
Predict Test Data → Save Results
```

---

## Feature Engineering

Six engineered features are derived from raw stats to capture player **efficiency** rather than just counting:

| Feature | Formula | Meaning |
|---|---|---|
| `Tackle_Success_Rate` | `Tackle Won / Tackle Attempt` | How often tackles succeed |
| `Shot_Accuracy` | `Shots on Target / Shots` | Shot precision |
| `Goal_Conversion` | `Goals / Shots` | Finishing efficiency |
| `Pass_Accuracy` | `Pass Completed / Pass Attempt` | Passing quality |
| `Goals_Per_App` | `Goals / Appearances` | Scoring rate per game |
| `Assists_Per_App` | `Assist / Appearances` | Assist rate per game |

> A small epsilon (`1e-6`) is added to all denominators to prevent division by zero.

---

## Model Selection

Two models were trained and compared:

| Model | Role | Notes |
|---|---|---|
| **Random Forest** | Baseline | Default parameters, parallel training |
| **Gradient Boosting** | Final Model | Tuned with GridSearchCV |

**Gradient Boosting was selected** as the final model due to higher validation accuracy.

**Hyperparameter tuning grid:**
```python
param_grid = {
    'gb__n_estimators':  [100, 200],
    'gb__learning_rate': [0.05, 0.1],
    'gb__max_depth':     [3, 4],
    'gb__subsample':     [0.8, 1.0]
}
```
Cross-validation: `StratifiedKFold(n_splits=5)`

---

## Results

The notebook includes:
- ✅ Accuracy comparison between Random Forest and Gradient Boosting
- ✅ Full Classification Report (precision, recall, F1-score per class)
- ✅ Confusion Matrix visualization
- ✅ Top 5 most important features per position (Striker, Midfielder, Defender, Goalkeeper)

**Key findings from feature importance analysis:**
- ⚽ **Striker** — `Goal_Conversion`, `Shot_Accuracy`, `Goals_Per_App`
- 🛡️ **Defender** — `Tackle_Success_Rate`, `Interception`, `Appearances`
- 🔄 **Midfielder** — `Pass_Accuracy`, `Assists_Per_App`, `Key Passes`
- 🧤 **Goalkeeper** — `Shutouts`, `Conceded`, `Appearances`

---

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/Nichonicholass/bigdata-soccer-player.git
cd bigdata-soccer-player
```

### 2. Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn gradio joblib
```

### 3. Run the Notebook

Open `soccer_analysis.ipynb` in Jupyter or VS Code. Run all cells to:
- Perform EDA and feature engineering
- Train and evaluate models
- Generate `soccer_pipeline.pkl` and `position_encoder.pkl`
- Predict test data

### 4. Launch the Web App (Optional)

```bash
python app.py
```

Then open the local URL shown in the terminal (e.g., `http://127.0.0.1:7860`).

> ⚠️ The `.pkl` model files must exist before running `app.py`. Run the notebook first.

---

## Web Application

`app.py` provides an interactive Gradio interface where you can:
- Input any player's statistics manually
- Get an instant performance prediction (`Bad` / `Normal` / `Good`)
- Works for all 4 positions: Striker, Midfielder, Defender, Goalkeeper

---

## Technologies Used

| Library | Purpose |
|---|---|
| `pandas` | Data manipulation |
| `numpy` | Numerical computation |
| `scikit-learn` | ML models, preprocessing, evaluation |
| `matplotlib` / `seaborn` | Data visualization |
| `gradio` | Interactive web UI |
| `joblib` | Model serialization (`.pkl`) |

---
