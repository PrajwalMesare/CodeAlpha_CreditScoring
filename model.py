"""
credit_scoring — model.py
Personal ML Project — Credit Scoring
Author : Prajwal Mesare
GitHub : github.com/PrajwalMesare

Standalone inference module.
Load the trained model and run predictions without re-training.

Usage
-----
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
"""

import os
import joblib
import numpy as np
import pandas as pd

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


class CreditScoringModel:
    """Wrapper for the trained credit scoring model."""

    def __init__(self):
        self.model         = joblib.load(os.path.join(MODEL_DIR, "credit_model.pkl"))
        self.scaler        = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        self.imputer       = joblib.load(os.path.join(MODEL_DIR, "imputer.pkl"))
        self.feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))

    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["TotalLatePayments"] = (
            df["NumberOfTime30-59DaysPastDueNotWorse"]
            + df["NumberOfTime60-89DaysPastDueNotWorse"]
            + df["NumberOfTimes90DaysLate"]
        )
        df["DebtPerDependent"] = df["DebtRatio"] / (df["NumberOfDependents"] + 1)
        df["IncomePerLoan"]    = df["MonthlyIncome"] / (df["NumberOfOpenCreditLinesAndLoans"] + 1)
        df["CreditUtilRisk"]   = df["RevolvingUtilizationOfUnsecuredLines"] * df["DebtRatio"]
        df["AgeGroup"]         = pd.cut(
            df["age"], bins=[0, 25, 35, 50, 65, 100], labels=[0, 1, 2, 3, 4]
        ).astype(int)
        return df

    def predict(self, input_data: dict) -> dict:
        """
        Predict creditworthiness for one applicant.

        Parameters
        ----------
        input_data : dict  — raw feature values

        Returns
        -------
        dict with keys:
            default_probability  : float  (0-1)
            prediction           : int    (0 or 1)
            risk_label           : str
            risk_level           : str    ('Low' / 'Medium' / 'High')
        """
        df = pd.DataFrame([input_data])
        df = self._engineer(df)
        df = df[self.feature_names]                       # ensure column order
        df_imp = pd.DataFrame(
            self.imputer.transform(df), columns=self.feature_names
        )

        proba      = self.model.predict_proba(df_imp)[0][1]
        prediction = int(proba > 0.5)

        if proba < 0.25:
            risk_level = "Low"
            risk_label = "✅ LOW RISK — Creditworthy"
        elif proba < 0.55:
            risk_level = "Medium"
            risk_label = "⚠️  MEDIUM RISK — Review Required"
        else:
            risk_level = "High"
            risk_label = "❌ HIGH RISK — Likely to Default"

        return {
            "default_probability": round(float(proba), 4),
            "prediction"         : prediction,
            "risk_label"         : risk_label,
            "risk_level"         : risk_level,
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run predictions on a DataFrame of applicants."""
        df_feat = self._engineer(df.copy())
        df_feat = df_feat[self.feature_names]
        df_imp  = pd.DataFrame(
            self.imputer.transform(df_feat), columns=self.feature_names
        )
        probas = self.model.predict_proba(df_imp)[:, 1]
        df["default_probability"] = probas.round(4)
        df["prediction"]          = (probas > 0.5).astype(int)
        df["risk_level"]          = pd.cut(
            probas, bins=[0, 0.25, 0.55, 1.0],
            labels=["Low", "Medium", "High"]
        )
        return df


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    m = CreditScoringModel()

    applicants = [
        {   # Low risk
            "RevolvingUtilizationOfUnsecuredLines": 0.15,
            "age": 48,
            "NumberOfTime30-59DaysPastDueNotWorse": 0,
            "DebtRatio": 0.18,
            "MonthlyIncome": 7500,
            "NumberOfOpenCreditLinesAndLoans": 9,
            "NumberOfTimes90DaysLate": 0,
            "NumberRealEstateLoansOrLines": 1,
            "NumberOfTime60-89DaysPastDueNotWorse": 0,
            "NumberOfDependents": 2,
        },
        {   # High risk
            "RevolvingUtilizationOfUnsecuredLines": 0.92,
            "age": 27,
            "NumberOfTime30-59DaysPastDueNotWorse": 4,
            "DebtRatio": 0.85,
            "MonthlyIncome": 1800,
            "NumberOfOpenCreditLinesAndLoans": 14,
            "NumberOfTimes90DaysLate": 2,
            "NumberRealEstateLoansOrLines": 0,
            "NumberOfTime60-89DaysPastDueNotWorse": 1,
            "NumberOfDependents": 3,
        },
    ]

    for i, appl in enumerate(applicants, 1):
        result = m.predict(appl)
        print(f"\nApplicant {i}")
        print(f"  Default Probability : {result['default_probability']*100:.2f}%")
        print(f"  Risk Level          : {result['risk_level']}")
        print(f"  Decision            : {result['risk_label']}")
