"""
Shared preprocessing utilities for training and inference.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

COLUMN_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]

BASE_FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

IMPUTE_FEATURES = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

ENGINEERED_FEATURES = [
    "Glucose_BMI_Interaction",
    "Age_BMI_Interaction",
    "Insulin_Glucose_Ratio",
    "RiskScore_Glucose_Age_BMI",
]

FEATURE_COLUMNS = BASE_FEATURES + ENGINEERED_FEATURES


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize schema artifacts and ensure numeric dtypes."""
    df_clean = df.copy()

    if len(df_clean) > 0 and str(df_clean.iloc[0]["Pregnancies"]) == "Pregnancies":
        df_clean = df_clean.iloc[1:].reset_index(drop=True)

    for col in COLUMN_NAMES:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    return df_clean.dropna().drop_duplicates().reset_index(drop=True)


def validate_input(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required base feature columns exist and are numeric."""
    missing = [c for c in BASE_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    for col in BASE_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if out[BASE_FEATURES].isnull().any().any():
        raise ValueError("Input contains non-numeric or missing values in base features.")

    return out


def fit_imputation_stats(df: pd.DataFrame) -> dict:
    """
    Compute overall median per imputable feature (zeros excluded).
    Used at inference when Outcome is unavailable.
    """
    stats = {}
    for feature in IMPUTE_FEATURES:
        non_zero = df.loc[df[feature] != 0, feature]
        stats[feature] = float(non_zero.median()) if len(non_zero) else 0.0
    return stats


def impute_zeros(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Replace zeros in medical features using saved median values."""
    out = df.copy()
    out[IMPUTE_FEATURES] = out[IMPUTE_FEATURES].astype(float)
    for feature in IMPUTE_FEATURES:
        mask = out[feature] == 0
        out.loc[mask, feature] = stats[feature]
    return out


def impute_zeros_stratified(df: pd.DataFrame) -> pd.DataFrame:
    """Replace zeros with per-class median values (training only)."""
    df_clean = df.copy()
    df_clean[IMPUTE_FEATURES] = df_clean[IMPUTE_FEATURES].astype(float)

    for feature in IMPUTE_FEATURES:
        for outcome_class in [0, 1]:
            mask_zero = df_clean[feature] == 0
            mask_class = df_clean["Outcome"] == outcome_class
            median_val = df_clean.loc[~mask_zero & mask_class, feature].median()
            df_clean.loc[mask_zero & mask_class, feature] = median_val

    return df_clean


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create clinically-inspired interaction and ratio features."""
    df_feat = df.copy()
    eps = 1e-6

    df_feat["Glucose_BMI_Interaction"] = df_feat["Glucose"] * df_feat["BMI"]
    df_feat["Age_BMI_Interaction"] = df_feat["Age"] * df_feat["BMI"]
    df_feat["Insulin_Glucose_Ratio"] = df_feat["Insulin"] / (df_feat["Glucose"] + eps)
    df_feat["RiskScore_Glucose_Age_BMI"] = (
        0.5 * df_feat["Glucose"] + 0.3 * df_feat["BMI"] + 0.2 * df_feat["Age"]
    )

    return df_feat


def transform_features(df: pd.DataFrame, scaler) -> pd.DataFrame:
    """Engineer features and apply fitted StandardScaler."""
    featured = engineer_features(validate_input(df))
    X = featured[FEATURE_COLUMNS]
    scaled = scaler.transform(X)
    columns = list(getattr(scaler, "feature_names_in_", FEATURE_COLUMNS))
    return pd.DataFrame(scaled, columns=columns, index=featured.index)


def preprocess_for_inference(
    df: pd.DataFrame, imputation_stats: dict, scaler
) -> pd.DataFrame:
    """Full inference path: validate -> impute zeros -> engineer -> scale."""
    validated = validate_input(df)
    imputed = impute_zeros(validated, imputation_stats)
    return transform_features(imputed, scaler)
