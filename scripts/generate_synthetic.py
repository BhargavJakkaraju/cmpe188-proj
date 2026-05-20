"""
Generate synthetic patient records for batch inference demo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing_utils import BASE_FEATURES, clean_dataset, impute_zeros_stratified
from src.inference import DiabetesRiskPredictor

LOCAL_DATA_PATH = PROJECT_ROOT / "data" / "pima-indians-diabetes.csv"
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
IMPUTE_FEATURES = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_reference_data() -> pd.DataFrame:
    df = pd.read_csv(LOCAL_DATA_PATH, names=COLUMN_NAMES)
    df = clean_dataset(df)
    return impute_zeros_stratified(df)


def generate_synthetic(
    reference: pd.DataFrame,
    n: int = 20,
    zero_rate: float = 0.08,
    random_state: int = 42,
    with_labels: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    rows = []

    for _ in range(n):
        row = {}
        for col in BASE_FEATURES:
            row[col] = float(rng.choice(reference[col].values))

        for feature in IMPUTE_FEATURES:
            if rng.random() < zero_rate:
                row[feature] = 0.0

        rows.append(row)

    synthetic = pd.DataFrame(rows)

    if with_labels:
        prevalence = reference["Outcome"].mean()
        synthetic["Outcome"] = rng.binomial(1, prevalence, size=n)

    return synthetic


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic patient data")
    parser.add_argument("--n", type=int, default=20, help="Number of synthetic patients")
    parser.add_argument(
        "--with-labels",
        action="store_true",
        help="Include sampled Outcome column for offline evaluation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/synthetic_data/synthetic_patients.csv",
    )
    parser.add_argument(
        "--predictions-output",
        type=str,
        default="results/synthetic_predictions.csv",
    )
    parser.add_argument("--no-predict", action="store_true", help="Skip inference step")
    args = parser.parse_args()

    reference = load_reference_data()
    synthetic = generate_synthetic(
        reference, n=args.n, with_labels=args.with_labels
    )

    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    synthetic.to_csv(out_path, index=False)
    print(f"Generated {len(synthetic)} synthetic patients -> {out_path}")

    if not args.no_predict:
        predictor = DiabetesRiskPredictor()
        results = predictor.predict_batch(out_path)
        pred_path = PROJECT_ROOT / args.predictions_output
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(pred_path, index=False)
        print(f"Saved predictions -> {pred_path}")


if __name__ == "__main__":
    main()
