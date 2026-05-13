# Formula 1 Pit Stop Prediction Using Machine Learning

Machine Learning based predictive analytics project developed for the Kaggle Playground Series 2026 competition.

---

# Overview

This project focuses on predicting whether a Formula 1 driver will make a pit stop on the next lap using machine learning algorithms and advanced feature engineering techniques.

The competition dataset is inspired by real-world Formula 1 strategy data and contains race-related information such as lap performance, tyre conditions, race positions, and driver behavior patterns.

The primary objective is to maximize the ROC-AUC score by building an optimized machine learning pipeline.

---

# Problem Statement

The task is to predict:

```python
PitNextLap
```

Where:

| Value | Meaning                         |
| ----- | ------------------------------- |
| 0     | No Pit Stop                     |
| 1     | Driver will pit on the next lap |

The challenge involves analyzing racing strategy, tyre degradation, pace variations, and sequential lap behavior.

---

# Competition Metric

The official evaluation metric is:

## ROC-AUC Score

A higher ROC-AUC score indicates better classification performance between pit-stop and non-pit-stop scenarios.

---

# Dataset Files

The competition dataset contains:

```bash
train.csv
test.csv
sample_submission.csv
```

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- CatBoost
- Kaggle Notebook Environment

---

# Project Workflow

## 1. Data Loading

- Load train and test datasets
- Inspect dataset structure
- Analyze feature types

---

## 2. Exploratory Data Analysis (EDA)

EDA techniques used:

- Histogram Analysis
- Countplots
- Correlation Heatmaps
- Feature Distribution Analysis
- Class Imbalance Analysis
- Missing Value Detection

---

## 3. Data Preprocessing

The preprocessing pipeline includes:

- Missing value handling
- Duplicate removal
- Label Encoding
- Numerical imputation
- Data sorting
- NaN handling

---

# Advanced Feature Engineering

Feature engineering is the most important component of this project.

The following advanced features were created:

---

## Lag Features

Historical race information from previous laps:

```python
feature_lag_1
feature_lag_2
feature_lag_3
feature_lag_5
```

These features help the model understand sequential race behavior.

---

## Rolling Window Features

Rolling statistics over multiple lap windows:

```python
rolling_mean
rolling_std
```

Window sizes used:

- 3
- 5
- 10

---

## Expanding Features

Cumulative historical statistics:

```python
expanding_mean
```

These help model long-term race pace trends.

---

## Difference Features

Difference between current and previous laps:

```python
current_value - previous_value
```

Useful for detecting sudden race strategy changes.

---

# Machine Learning Model

The final solution uses:

## CatBoost Classifier

Why CatBoost?

- Excellent for tabular datasets
- Handles complex relationships
- Reduces overfitting
- Strong performance on Kaggle competitions
- Works well with engineered features

---

# Cross Validation Strategy

The project uses:

```python
StratifiedKFold(n_splits=10)
```

Benefits:

- Better generalization
- Stable validation scores
- Reduced overfitting risk

---

# Model Parameters

```python
CatBoostClassifier(
    iterations=2000,
    learning_rate=0.02,
    depth=6,
    eval_metric="AUC"
)
```

---

# Performance

| Model               | ROC-AUC          |
| ------------------- | ---------------- |
| Logistic Regression | Baseline         |
| Random Forest       | ~0.93            |
| CatBoost            | High Performance |

The final optimized pipeline focuses on maximizing leaderboard performance using advanced feature engineering.

---

# Feature Importance

The model analyzes the most influential features affecting pit stop decisions, including:

- Lap trends
- Pace variations
- Historical lap behavior
- Sequential race dynamics
- Rolling statistics

---

# Submission File

Final predictions are saved as:

```bash
submission.csv
```

---

# How to Run

## Step 1

Clone repository:

```bash
git clone https://github.com/ShivamMathtech/formual_1_kagg_pit_comp
```

---

## Step 2

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn catboost
```

---

## Step 3

Run notebook:

```bash
jupyter notebook
```

---

# Project Structure

```bash
Formula1-PitStop-Prediction/
│
├── train.csv
├── test.csv
├── sample_submission.csv
├── notebook.ipynb
├── submission.csv
└── README.md
```

---

# Future Improvements

Possible future enhancements:

- LightGBM Ensemble
- XGBoost Stacking
- Deep Learning Models
- LSTM Sequential Models
- Reinforcement Learning for Race Strategy
- Real-time Pit Stop Analytics

---

# Key Learnings

This project demonstrates:

- Advanced Feature Engineering
- Sequential Data Processing
- Tabular Machine Learning
- Kaggle Competition Workflow
- Cross Validation Strategies
- Ensemble Modeling Techniques

---

# Author

Your Name

Department of Computer Science Engineering

May 2026

---

# References

- Kaggle Playground Series 2026
- CatBoost Documentation
- Scikit-Learn Documentation
- Formula 1 Strategy Analytics
- Machine Learning for Tabular Data
