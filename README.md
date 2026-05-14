# Credit Card Fraud Detection Model

A machine learning pipeline for detecting fraudulent credit card transactions on an imbalanced dataset of 1.8M records. Compares Logistic Regression, Decision Tree, and Random Forest classifiers with SMOTE resampling, hyperparameter tuning, and threshold optimization.

---

## Overview

Credit card fraud detection is a classic imbalanced classification problem — fraudulent transactions are rare, so models trained naively tend to ignore them entirely. This project addresses that through:

- **SMOTE** to synthetically oversample the minority class
- **Three baseline models** compared head-to-head
- **Randomized hyperparameter search** on the best-performing model
- **Threshold tuning** to optimize for business priorities (catching fraud over minimizing false alarms)

---

## Dataset

- **Train set:** ~1.8M transactions
- **Test set:** held-out split
- **Target:** `is_fraud` (binary: 0 = legitimate, 1 = fraud)
- **Class imbalance:** heavily skewed toward legitimate transactions

Features include transaction amount, timestamp, merchant category, cardholder demographics, and geographic coordinates.

---

## Project Structure

```
├── train.csv
├── test.csv
├── fraud_detection.ipynb      # Main notebook
└── README.md
```

---

## Pipeline

### 1. Feature Engineering
- Extracted `trans_hour`, `trans_dayofweek`, `trans_month` from timestamp
- Derived `age` from date of birth
- Computed `distance_km` between cardholder and merchant using Haversine formula

### 2. EDA
- Fraud rate by category, hour, age group, and state
- Correlation matrix to identify redundant features
- Average transaction amount: fraud vs. legitimate

### 3. Preprocessing
- Dropped high-cardinality and redundant columns (e.g. `merchant`, `zip`, raw coordinates)
- Label encoded categorical features (`category`, `gender`, `state`)
- Applied **SMOTE** to balance training classes before modelling

### 4. Baseline Models
| Model | Precision | Recall | F1 |
|---|---|---|---|
| Logistic Regression | — | — | — |
| Decision Tree | — | — | — |
| Random Forest | — | — | — |

> Fill in your actual scores from the classification reports

### 5. Hyperparameter Tuning
- Used `RandomizedSearchCV` with 3-fold stratified CV on 30% of resampled training data (for compute efficiency)
- Tuned: `max_depth`, `min_samples_split`, `min_samples_leaf`, `n_estimators`

### 6. Threshold Tuning
- Generated precision-recall curve on tuned Random Forest
- Identified mathematically optimal threshold via F1 maximization
- Applied **business threshold of 0.30** to prioritize recall (catching fraud matters more than false alarm rate)

**Final Model — Random Forest (threshold = 0.30):**
- F1 Score: **0.69**
- False alarm rate reduced by **77%** vs. baseline

---

## Key Findings

- Fraud rate varies significantly by merchant category and hour of day
- Distance between cardholder and merchant was not a strong fraud signal
- SMOTE + threshold tuning meaningfully improved recall without collapsing precision
- Random Forest outperformed both Logistic Regression and Decision Tree across all metrics

---

## Tech Stack

- **Python** — Pandas, NumPy, scikit-learn, imbalanced-learn
- **Visualization** — Matplotlib, Seaborn, Plotly
- **Environment** — Jupyter Notebook / Google Colab

---

## How to Run

```bash
# Install dependencies
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn plotly kaleido

# Launch notebook
jupyter notebook fraud_detection.ipynb
```

Ensure `train.csv` and `test.csv` are in the same directory as the notebook.

---
