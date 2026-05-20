"""
Inference module for diabetes risk prediction.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.preprocessing_utils import (
    BASE_FEATURES,
    FEATURE_COLUMNS,
    preprocess_for_inference,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "results" / "xgboost.pkl"
DEFAULT_SCALER_PATH = PROJECT_ROOT / "results" / "scaler.pkl"
DEFAULT_IMPUTATION_PATH = PROJECT_ROOT / "results" / "imputation_stats.pkl"


class DiabetesRiskPredictor:
    """Load artifacts and run single or batch predictions."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        scaler_path: str | Path | None = None,
        imputation_path: str | Path | None = None,
    ):
        self.model_path = Path(
            model_path or os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)
        )
        self.scaler_path = Path(
            scaler_path or os.environ.get("SCALER_PATH", DEFAULT_SCALER_PATH)
        )
        self.imputation_path = Path(
            imputation_path
            or os.environ.get("IMPUTATION_PATH", DEFAULT_IMPUTATION_PATH)
        )

        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        self.imputation_stats = joblib.load(self.imputation_path)

    def _to_dataframe(self, patient: dict | pd.DataFrame) -> pd.DataFrame:
        if isinstance(patient, dict):
            return pd.DataFrame([patient])
        return patient.copy()

    def _prepare_features(self, patient: dict | pd.DataFrame) -> pd.DataFrame:
        df = self._to_dataframe(patient)
        return preprocess_for_inference(df, self.imputation_stats, self.scaler)

    def predict_proba(self, patient: dict | pd.DataFrame) -> np.ndarray:
        X = self._prepare_features(patient)
        return self.model.predict_proba(X)

    def predict(self, patient: dict | pd.DataFrame) -> pd.DataFrame:
        X = self._prepare_features(patient)
        probs = self.model.predict_proba(X)
        preds = self.model.predict(X)

        results = []
        for i, pred in enumerate(preds):
            prob_diabetic = float(probs[i, 1])
            prob_non_diabetic = float(probs[i, 0])
            confidence = max(prob_diabetic, prob_non_diabetic)
            results.append(
                {
                    "prediction": int(pred),
                    "risk_label": "Diabetic" if pred == 1 else "Non-Diabetic",
                    "probability_diabetic": prob_diabetic,
                    "probability_non_diabetic": prob_non_diabetic,
                    "confidence": confidence,
                }
            )

        return pd.DataFrame(results)

    def explain_features(self, top_n: int = 8) -> pd.DataFrame:
        """Return global feature importances from the loaded model."""
        if not hasattr(self.model, "feature_importances_"):
            return pd.DataFrame(columns=["feature", "importance"])

        importances = self.model.feature_importances_
        columns = list(getattr(self.scaler, "feature_names_in_", FEATURE_COLUMNS))
        fi = pd.DataFrame({"feature": columns, "importance": importances})
        return fi.sort_values("importance", ascending=False).head(top_n).reset_index(
            drop=True
        )

    def predict_batch(self, csv_path: str | Path) -> pd.DataFrame:
        """Run predictions on a CSV of base features."""
        csv_path = Path(csv_path)
        batch = pd.read_csv(csv_path)
        preds = self.predict(batch)
        input_cols = [c for c in batch.columns if c in BASE_FEATURES]
        return pd.concat([batch[input_cols].reset_index(drop=True), preds], axis=1)

    @staticmethod
    def default_input_values(imputation_stats: dict) -> dict:
        """Suggested defaults for UI forms (training medians)."""
        defaults = {
            "Pregnancies": 3.0,
            "DiabetesPedigreeFunction": 0.47,
            "Age": 33.0,
        }
        for feature in imputation_stats:
            defaults[feature] = imputation_stats[feature]
        return defaults


def _parse_args():
    parser = argparse.ArgumentParser(description="Diabetes risk inference CLI")
    parser.add_argument("--input", type=str, help="Path to single-patient JSON file")
    parser.add_argument("--csv", type=str, help="Path to batch CSV of patients")
    parser.add_argument(
        "--output",
        type=str,
        default="results/synthetic_predictions.csv",
        help="Output path for batch predictions",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    predictor = DiabetesRiskPredictor()

    if args.csv:
        results = predictor.predict_batch(args.csv)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(out_path, index=False)
        print(f"Saved {len(results)} predictions to {out_path}")
        print(results.head())
        return

    if args.input:
        with open(args.input, encoding="utf-8") as f:
            patient = json.load(f)
        result = predictor.predict(patient)
        print(result.to_string(index=False))
        return

    print("Provide --input <patient.json> or --csv <patients.csv>")


if __name__ == "__main__":
    main()
