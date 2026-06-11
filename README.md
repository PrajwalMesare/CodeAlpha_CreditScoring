# 🏦 Credit Scoring Model

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)

**Personal ML Project**

**Developer:** Prajwal Mesare | TGPCET Nagpur | B.Tech CSE (Data Science) 2027

[![GitHub](https://img.shields.io/badge/GitHub-PrajwalMesare-181717?logo=github)](https://github.com/PrajwalMesare)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin)](https://linkedin.com/in/prajwal-mesare-700678263)

</div>

---

## 📌 Objective

Predict whether a borrower will experience **serious delinquency (90+ days past due)** within 2 years using the real-world **Give Me Some Credit** dataset from Kaggle.

---

## 📊 Dataset

| Field | Detail |
|-------|--------|
| **Source** | [Kaggle — Give Me Some Credit](https://www.kaggle.com/competitions/GiveMeSomeCredit) |
| **Rows** | 150,000 borrowers |
| **Features** | 10 financial & behavioral features |
| **Target** | `SeriousDlqin2yrs` — 1 = Default, 0 = No Default |
| **Default Rate** | ~10.8% (class imbalanced → fixed with SMOTE) |

---

## 🗂️ Project Structure

```
Credit-Scoring-Model/
│
├── credit_scoring_model.ipynb   ← Full ML pipeline (EDA → Train → Evaluate)
├── model.py                     ← Standalone inference module
├── requirements.txt
├── .gitignore
├── README.md
│
├── data/
│   └── cs-training.csv          ← Kaggle dataset (150K rows)
│
├── models/
│   ├── credit_model.pkl         ← Trained XGBoost model
│   ├── scaler.pkl               ← StandardScaler
│   ├── imputer.pkl              ← SimpleImputer
│   └── feature_names.pkl        ← Feature column order
│
└── outputs/
    ├── eda_plots.png            ← EDA visualizations
    └── model_evaluation.png     ← ROC, confusion matrix, feature importance
```

---

## 🔄 ML Pipeline

```
Load CSV (150K rows)
    ↓
Handle Missing Values (MonthlyIncome, NumberOfDependents)
    ↓
Cap Outliers (99th percentile)
    ↓
Feature Engineering
  → TotalLatePayments
  → DebtPerDependent
  → IncomePerLoan
  → CreditUtilRisk
  → AgeGroup
    ↓
Train / Test Split (80/20, stratified)
    ↓
SMOTE (sampling_strategy=0.3 to balance classes)
    ↓
Train 5 Models
    ↓
Evaluate (ROC-AUC, F1, Accuracy, Precision, Recall)
    ↓
Export Best Model
```

---

## 📈 Results

| Model | ROC-AUC | F1-Score | Accuracy |
|-------|:-------:|:--------:|:--------:|
| Logistic Regression | 0.7034 | 0.1463 | 0.8943 |
| Decision Tree | 0.7424 | 0.0180 | 0.9179 |
| Random Forest | 0.7493 | 0.0043 | 0.9236 |
| Gradient Boosting | 0.7537 | 0.0049 | 0.9241 |
| **XGBoost ✅** | **0.7538** | **0.0037** | **0.9243** |

> **Best Model: XGBoost** — ROC-AUC = **0.7538**

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Notebook
```bash
jupyter notebook credit_scoring_model.ipynb
```
> Run all cells — this trains all models, saves outputs, and exports the best model.

### 3. Use the Inference Module
```python
from model import CreditScoringModel

m = CreditScoringModel()
result = m.predict({
    "RevolvingUtilizationOfUnsecuredLines": 0.35,
    "age": 42,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.25,
    "MonthlyIncome": 5000,
    "NumberOfOpenCreditLinesAndLoans": 8,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 1
})

print(result)
# {'default_probability': 0.12, 'risk_level': 'Low', 'risk_label': '✅ LOW RISK — Creditworthy'}
```

---

## 📊 EDA Plots

| Plot | Description |
|------|-------------|
| `outputs/eda_plots.png` | Target distribution, age histogram, revolving utilization, debt ratio boxplot, late payment correlations, income distribution |
| `outputs/model_evaluation.png` | ROC curves (5 models), confusion matrix, model comparison bar chart, feature importance |

---

## 🛠️ Tech Stack

`Python` · `pandas` · `numpy` · `scikit-learn` · `XGBoost` · `imbalanced-learn` · `matplotlib` · `seaborn` · `joblib` · `Jupyter`

---

## 📜 About

Built as a personal ML project for portfolio and learning purposes.  
