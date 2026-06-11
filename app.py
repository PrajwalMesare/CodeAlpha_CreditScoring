"""
app.py — Task 1: Credit Scoring Model
Personal ML Project
Author: Prajwal Mesare | github.com/PrajwalMesare
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Scoring Model",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2.2rem 2rem; border-radius: 16px; margin-bottom: 1.5rem;
    color: white; text-align: center;
}
.header h1 { font-size: 2.2rem; font-weight: 700; margin: 0; }
.header p  { color: #a8b4c8; font-size: 1rem; margin-top: 0.4rem; }

.card {
    background: #f8f9fc; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 1.4rem; margin-bottom: 1rem;
}
.metric-box {
    background: linear-gradient(135deg, #667eea, #764ba2);
    border-radius: 12px; padding: 1.2rem; color: white; text-align: center;
}
.metric-box .val { font-size: 1.9rem; font-weight: 700; }
.metric-box .lbl { font-size: 0.82rem; opacity: 0.85; }

.risk-low    { background: linear-gradient(135deg,#11998e,#38ef7d);
               padding:1.2rem; border-radius:12px; color:white;
               text-align:center; font-size:1.2rem; font-weight:700; }
.risk-medium { background: linear-gradient(135deg,#f7971e,#ffd200);
               padding:1.2rem; border-radius:12px; color:#1a1a2e;
               text-align:center; font-size:1.2rem; font-weight:700; }
.risk-high   { background: linear-gradient(135deg,#eb3349,#f45c43);
               padding:1.2rem; border-radius:12px; color:white;
               text-align:center; font-size:1.2rem; font-weight:700; }

.info { background:#ebf8ff; border-left:4px solid #3182ce;
        padding:0.8rem 1rem; border-radius:0 8px 8px 0;
        font-size:0.9rem; color:#2c5282; margin-bottom:1rem; }

.stButton>button {
    background: linear-gradient(135deg,#667eea,#764ba2);
    color:white; border:none; border-radius:8px;
    padding:0.65rem 2rem; font-weight:600; font-size:1rem;
    width:100%; cursor:pointer; transition: opacity 0.2s;
}
.stButton>button:hover { opacity: 0.88; }
</style>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model         = joblib.load("models/credit_model.pkl")
    scaler        = joblib.load("models/scaler.pkl")
    imputer       = joblib.load("models/imputer.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    return model, scaler, imputer, feature_names

def engineer_features(df):
    df = df.copy()
    df["TotalLatePayments"] = (df["NumberOfTime30-59DaysPastDueNotWorse"] +
                                df["NumberOfTime60-89DaysPastDueNotWorse"] +
                                df["NumberOfTimes90DaysLate"])
    df["DebtPerDependent"]  = df["DebtRatio"] / (df["NumberOfDependents"] + 1)
    df["IncomePerLoan"]     = df["MonthlyIncome"] / (df["NumberOfOpenCreditLinesAndLoans"] + 1)
    df["CreditUtilRisk"]    = df["RevolvingUtilizationOfUnsecuredLines"] * df["DebtRatio"]
    df["AgeGroup"]          = pd.cut(df["age"], bins=[0,25,35,50,65,100],
                                      labels=[0,1,2,3,4]).astype(int)
    return df

def predict(inputs, model, scaler, imputer, feature_names):
    df     = pd.DataFrame([inputs])
    df     = engineer_features(df)
    df     = df[feature_names]
    df_imp = pd.DataFrame(imputer.transform(df), columns=feature_names)
    proba  = model.predict_proba(df_imp)[0][1]
    if proba < 0.25:
        level, label = "low",    "✅ LOW RISK — Creditworthy"
    elif proba < 0.55:
        level, label = "medium", "⚠️ MEDIUM RISK — Review Required"
    else:
        level, label = "high",   "❌ HIGH RISK — Likely to Default"
    return proba, level, label

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Credit Scoring")
    st.markdown("**Personal ML Project**")
    st.markdown("**Developer:** Prajwal Mesare")
    st.markdown("**Model:** XGBoost")
    st.markdown("**ROC-AUC:** 0.7538")
    st.markdown("---")
    st.markdown("**Dataset:** Kaggle — Give Me Some Credit")
    st.markdown("**Rows:** 150,000 borrowers")
    st.markdown("**Features:** 10 financial features")
    st.markdown("---")
    page = st.radio("Navigate", ["🔍 Predict", "📊 Model Performance", "📈 EDA Plots"])
    st.markdown("---")
    st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-PrajwalMesare-181717?logo=github)](https://github.com/PrajwalMesare)")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header">
    <h1>🏦 Credit Scoring Model</h1>
    <p>Kaggle — Give Me Some Credit | Predict Borrower Default Risk | Personal ML Project</p>
</div>
""", unsafe_allow_html=True)

# ── Metrics Row ───────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="metric-box"><div class="val">150K</div><div class="lbl">Training Rows</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-box"><div class="val">0.754</div><div class="lbl">Best ROC-AUC</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-box"><div class="val">5</div><div class="lbl">Models Trained</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="metric-box"><div class="val">XGBoost</div><div class="lbl">Best Model</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Predict":
    st.markdown("### Enter Applicant Financial Details")
    st.markdown('<div class="info">Fill in the applicant\'s financial data based on the Kaggle <b>Give Me Some Credit</b> dataset features.</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**👤 Personal Info**")
        age        = st.slider("Age", 18, 80, 42)
        dependents = st.slider("Number of Dependents", 0, 10, 1)
        income     = st.number_input("Monthly Income ($)", 0, 50000, 5000, step=250)

    with col2:
        st.markdown("**💳 Credit Info**")
        revolving   = st.slider("Revolving Utilization (0–1)", 0.0, 1.0, 0.35, 0.01)
        open_credit = st.slider("Open Credit Lines & Loans", 0, 30, 8)
        real_estate = st.slider("Real Estate Loans", 0, 10, 1)
        debt_ratio  = st.slider("Debt Ratio", 0.0, 2.0, 0.25, 0.01)

    with col3:
        st.markdown("**⚠️ Payment History**")
        late_30_59 = st.slider("Times 30–59 Days Late", 0, 10, 0)
        late_60_89 = st.slider("Times 60–89 Days Late", 0, 10, 0)
        late_90    = st.slider("Times 90+ Days Late",   0, 10, 0)

    st.markdown("---")
    if st.button("🔮 Predict Credit Risk"):
        try:
            model, scaler, imputer, feature_names = load_model()
            inp = {
                "RevolvingUtilizationOfUnsecuredLines":      revolving,
                "age":                                        age,
                "NumberOfTime30-59DaysPastDueNotWorse":      late_30_59,
                "DebtRatio":                                  debt_ratio,
                "MonthlyIncome":                              income,
                "NumberOfOpenCreditLinesAndLoans":            open_credit,
                "NumberOfTimes90DaysLate":                    late_90,
                "NumberRealEstateLoansOrLines":               real_estate,
                "NumberOfTime60-89DaysPastDueNotWorse":      late_60_89,
                "NumberOfDependents":                         dependents,
            }
            proba, level, label = predict(inp, model, scaler, imputer, feature_names)
            prob_pct = proba * 100

            st.markdown("### 📋 Prediction Result")
            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown(f'<div class="risk-{level}">{label}</div>', unsafe_allow_html=True)
            with r2:
                st.metric("Default Probability", f"{prob_pct:.1f}%")
            with r3:
                st.metric("Risk Level", level.upper())

            # Gauge bar
            st.markdown("<br>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(8, 2))
            color = "#11998e" if level == "low" else ("#f7971e" if level == "medium" else "#eb3349")
            ax.barh([""], [prob_pct],   color=color,    height=0.5)
            ax.barh([""], [100 - prob_pct], left=[prob_pct], color="#e2e8f0", height=0.5)
            ax.set_xlim(0, 100)
            ax.axvline(25, color="green",  linestyle="--", linewidth=1, alpha=0.6)
            ax.axvline(55, color="orange", linestyle="--", linewidth=1, alpha=0.6)
            ax.set_xlabel("Default Probability (%)", fontsize=11)
            ax.set_title(f"Default Probability: {prob_pct:.1f}%  |  Risk: {level.upper()}", fontsize=13, fontweight="bold")
            ax.spines[['top','right','left']].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig); plt.close()

            # Input summary
            with st.expander("📄 View Input Summary"):
                summary = pd.DataFrame([{
                    "Age": age, "Monthly Income": f"${income:,}",
                    "Debt Ratio": debt_ratio, "Revolving Utilization": revolving,
                    "Open Credit Lines": open_credit, "Late 30-59 Days": late_30_59,
                    "Late 60-89 Days": late_60_89, "Late 90+ Days": late_90,
                    "Real Estate Loans": real_estate, "Dependents": dependents,
                }]).T.rename(columns={0: "Value"})
                st.dataframe(summary, use_container_width=True)

        except Exception as e:
            st.error(f"Model not found. Run the notebook first to generate models/. Error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.markdown("### 📊 Model Comparison — All 5 Algorithms")

    perf = pd.DataFrame({
        "Model":     ["Logistic Regression","Decision Tree","Random Forest","Gradient Boosting","XGBoost ✅"],
        "ROC-AUC":   [0.7034, 0.7424, 0.7493, 0.7537, 0.7538],
        "F1-Score":  [0.1463, 0.0180, 0.0043, 0.0049, 0.0037],
        "Accuracy":  [0.8943, 0.9179, 0.9236, 0.9241, 0.9243],
        "Precision": [0.5120, 0.3500, 0.4200, 0.4100, 0.3900],
        "Recall":    [0.0810, 0.0094, 0.0022, 0.0025, 0.0019],
    })
    st.dataframe(
        perf.style.highlight_max(subset=["ROC-AUC","Accuracy"], color="#d4edda")
                  .highlight_max(subset=["F1-Score"], color="#cce5ff")
                  .format({"ROC-AUC":"{:.4f}","F1-Score":"{:.4f}",
                           "Accuracy":"{:.4f}","Precision":"{:.4f}","Recall":"{:.4f}"}),
        use_container_width=True
    )

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    x   = np.arange(len(perf["Model"]))
    w   = 0.25
    b1  = ax.bar(x - w, perf["ROC-AUC"],  w, label="ROC-AUC",  color="#3498db", edgecolor="white")
    b2  = ax.bar(x,     perf["Accuracy"], w, label="Accuracy",  color="#2ecc71", edgecolor="white")
    b3  = ax.bar(x + w, perf["F1-Score"], w, label="F1-Score",  color="#e74c3c", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(perf["Model"], rotation=15, ha="right", fontsize=10)
    ax.set_ylim(0, 1.05); ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.legend(); ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("""
    <div class="card">
    <b>Best Model: XGBoost</b><br><br>
    The dataset has a <b>~10.8% default rate</b> (class imbalanced), fixed using SMOTE oversampling.<br>
    ROC-AUC is the primary metric here as it best handles imbalanced binary classification.<br><br>
    <b>Key engineered features:</b> TotalLatePayments, DebtPerDependent, IncomePerLoan, CreditUtilRisk, AgeGroup
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA PLOTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 EDA Plots":
    st.markdown("### 📈 Exploratory Data Analysis")

    eda_path  = "outputs/eda_plots.png"
    eval_path = "outputs/model_evaluation.png"

    if os.path.exists(eda_path):
        st.image(eda_path, caption="Give Me Some Credit — EDA (Target, Age, Utilization, Debt, Correlations, Income)", use_column_width=True)
    else:
        st.warning("EDA plot not found. Run the notebook first.")

    st.markdown("---")

    if os.path.exists(eval_path):
        st.image(eval_path, caption="Model Evaluation — ROC Curves, Confusion Matrix, Comparison, Feature Importance", use_column_width=True)
    else:
        st.warning("Evaluation plot not found. Run the notebook first.")
